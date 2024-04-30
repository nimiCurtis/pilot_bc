from typing import List, Dict, Optional, Tuple
from pilot_models.policy.base_model import BaseModel
from pilot_models.encoder.model_registry import get_encoder_model
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=6):
        super().__init__()

        # Compute the positional encoding once
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)

        # Register the positional encoding as a buffer to avoid it being
        # considered a parameter when saving the model
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        # Add the positional encoding to the input
        x = x + self.pos_enc[:, :x.size(1), :]
        return x

class MultiLayerDecoder(nn.Module):
    def __init__(self, embed_dim=512, seq_len=6, output_layers=[256, 128, 64], nhead=8, num_layers=8, ff_dim_factor=4):
        super(MultiLayerDecoder, self).__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len=seq_len)
        self.sa_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim_factor*embed_dim, activation="gelu", batch_first=True, norm_first=True)
        self.sa_decoder = nn.TransformerEncoder(self.sa_layer, num_layers=num_layers)
        self.output_layers = nn.ModuleList([nn.Linear(seq_len*embed_dim, embed_dim)])
        self.output_layers.append(nn.Linear(embed_dim, output_layers[0]))
        for i in range(len(output_layers)-1):
            self.output_layers.append(nn.Linear(output_layers[i], output_layers[i+1]))

    def forward(self, x):
        if self.positional_encoding: x = self.positional_encoding(x)
        x = self.sa_decoder(x)
        # currently, x is [batch_size, seq_len, embed_dim]
        x = x.reshape(x.shape[0], -1)
        for i in range(len(self.output_layers)):
            x = self.output_layers[i](x)
            x = F.relu(x)
        return x

class ViNT(BaseModel):
    def __init__(
            self,
            policy_model_cfg: DictConfig,
            encoder_model_cfg: DictConfig,
            training_cfg: DictConfig,
            data_cfg: DictConfig 
    ) -> None:
        """
        ViNT class: uses a Transformer-based architecture to encode (current and past) visual observations
        and goals using an EfficientNet CNN, and predicts temporal distance and normalized actions
        in an embodiment-agnostic manner
        Args:
            context_size (int): how many previous observations to used for context
            len_traj_pred (int): how many waypoints to predict in the future
            learn_angle (bool): whether to predict the yaw of the robot
            obs_encoder (str): name of the EfficientNet architecture to use for encoding observations (ex. "efficientnet-b0")
            obs_encoding_size (int): size of the encoding of the observation images
            goal_encoding_size (int): size of the encoding of the goal images
        """
        
        # Data config
        context_size=data_cfg.context_size
        len_traj_pred=data_cfg.len_traj_pred
        self.learn_angle=data_cfg.learn_angle
        
        # Policy model
        mha_num_attention_heads=policy_model_cfg.mha_num_attention_heads
        mha_num_attention_layers=policy_model_cfg.mha_num_attention_layers
        mha_ff_dim_factor=policy_model_cfg.mha_ff_dim_factor
        self.obs_encoding_size=policy_model_cfg.obs_encoding_size
        self.late_fusion=policy_model_cfg.late_fusion

        # Training config
        self.goal_condition=training_cfg.goal_condition
        self.alpha = training_cfg.alpha
        self.beta = training_cfg.beta
        
        super(ViNT, self).__init__(policy_model_cfg.name,
                                context_size,
                                len_traj_pred,
                                self.learn_angle,
                                encoder_model_cfg.in_channels)
        
        seq_len = self.context_size + 2 if self.goal_condition else self.context_size + 1
        
        self.obs_encoder = get_encoder_model(encoder_model_cfg)
        self.obs_encoder = replace_bn_with_gn(self.obs_encoder)
        # linear input encoder

        lin_encoding_size = policy_model_cfg.lin_encoding_size  # should match obs_encoding_size for easy concat
        num_lin_features = policy_model_cfg.num_lin_features  # sum of features in current_rel_pos_to_target & goal_rel_pos_to_obj

        # think of a better encoding
        self.lin_encoder = nn.Sequential(nn.Linear(num_lin_features, lin_encoding_size // 2),
                                         nn.ReLU(),
                                         nn.Linear(lin_encoding_size // 2, lin_encoding_size))

        self.num_obs_features = self.obs_encoder.get_in_feateures()
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()

        self.decoder = MultiLayerDecoder(
            embed_dim=self.obs_encoding_size,
            seq_len=seq_len,
            output_layers=[256, 128, 64, 32],
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
        )
        self.dist_predictor = nn.Sequential(
            nn.Linear(32, 1),
        )
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * self.num_action_params),
        )

    def forward(
            self, obs_img: torch.tensor, current_rel_pos_to_target: torch.tensor, goal_rel_pos_to_obj: torch.tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # split the observation into context based on the context size
        # Currently obs_img size is [batch_size, C*(self.context_size+1), H, W] | for example: [16, 3*(5+1), 64, 85]
        obs_img = torch.split(obs_img, self.in_channels, dim=1)

        # Currently obs_img size is [self.context_size+1, batch_size, C, H, W] | for example: [6, 16, 3, 64, 85]
        obs_img = torch.concat(obs_img, dim=0)

        # Currently obs_img size is [batch_size*(self.context_size+1), C, H, W] | for example: [96, 3, 64, 85]
        obs_encoding = self.obs_encoder.extract_features(obs_img)
        
        # (Encoded) Currently obs_encoding size is [batch_size*(self.context_size+1), source_encoder_out_features] | for example: [96, 1280]
        obs_encoding = self.compress_obs_enc(obs_encoding)
        
        # (Compressed) Currently obs_encoding size is [batch_size*(self.context_size + 1), self.obs_encoding_size (= a param from config)] | for example: [96, 512]
        obs_encoding = obs_encoding.reshape((self.context_size + 1, -1, self.obs_encoding_size))
        # (reshaped) Currently obs_encoding is [self.context_size + 1, batch_size, self.obs_encoding_size], note that the order is flipped | for example: [6, 16, 512]
        obs_encoding = torch.transpose(obs_encoding, 0, 1)
        # (transposed) Currently obs_encoding size is [batch_size, self.context_size+1, self.obs_encoding_size] | for example: [16, 6, 512]

        if self.goal_condition:
            linear_input = torch.cat((current_rel_pos_to_target, goal_rel_pos_to_obj), dim=1)
        else:
            linear_input = current_rel_pos_to_target
        lin_encoding = self.lin_encoder(linear_input)
        if len(lin_encoding.shape) == 2:
            lin_encoding = lin_encoding.unsqueeze(1)
        # currently, the size of goal_encoding is [batch_size, 1, self.goal_encoding_size]
        assert lin_encoding.shape[2] == self.lin_encoding_size

        tokens = torch.cat((obs_encoding, lin_encoding), dim=1)  # obs_encoding
        final_repr = self.decoder(tokens)

        # currently, the size is [batch_size, 32]
        dist_pred = self.dist_predictor(final_repr)
        action_pred = self.action_predictor(final_repr)

        # augment outputs to match labels size-wise
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )

        action_pred[:, :, :2] = torch.cumsum(
            action_pred[:, :, :2], dim=1
        )  # convert position deltas into waypoints
        if self.learn_angle:
            action_pred[:, :, 2:] = F.normalize(
                action_pred[:, :, 2:].clone(), dim=-1
            )  # normalize the angle prediction
        return dist_pred, action_pred
    
    def _compute_losses(
        self,
            dist_label: torch.Tensor,
            action_label: torch.Tensor,
            dist_pred: torch.Tensor,
            action_pred: torch.Tensor,
            action_mask: torch.Tensor = None,
    ):
        """
        Compute losses for distance and action prediction.

        """
        dist_loss = F.mse_loss(dist_pred.squeeze(-1), dist_label.float())

        def action_reduce(unreduced_loss: torch.Tensor):
            # Reduce over non-batch dimensions to get loss per batch element
            while unreduced_loss.dim() > 1:
                unreduced_loss = unreduced_loss.mean(dim=-1)
            assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
            return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

        # Mask out invalid inputs (for negatives, or when the distance between obs and goal is large)
        # This is the actual losses
        assert action_pred.shape == action_label.shape, f"{action_pred.shape} != {action_label.shape}"
        action_loss = action_reduce(F.mse_loss(action_pred, action_label, reduction="none"))

        # Other losses for logger
        action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
            action_pred[:, :, :2], action_label[:, :, :2], dim=-1
        ))
        multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
            torch.flatten(action_pred[:, :, :2], start_dim=1),
            torch.flatten(action_label[:, :, :2], start_dim=1),
            dim=-1,
        ))

        results = {
            "dist_loss": dist_loss,
            "action_loss": action_loss,
            "action_waypts_cos_sim": action_waypts_cos_similairity,
            "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim,
        }

        if self.learn_angle:
            action_orien_cos_sim = action_reduce(F.cosine_similarity(
                action_pred[:, :, 2:], action_label[:, :, 2:], dim=-1
            ))
            multi_action_orien_cos_sim = action_reduce(F.cosine_similarity(
                torch.flatten(action_pred[:, :, 2:], start_dim=1),
                torch.flatten(action_label[:, :, 2:], start_dim=1),
                dim=-1,
            )
            )
            results["action_orien_cos_sim"] = action_orien_cos_sim
            results["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim

        total_loss = self.alpha * 1e-2 * dist_loss + self.beta * (1 - self.alpha) * action_loss
        results["total_loss"] = total_loss

        return results

from typing import List, Dict, Optional, Tuple, Callable
# Utils for Group Norm
def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module