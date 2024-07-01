from typing import Tuple
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.nn.functional as F

from pilot_models.policy.base_model import BaseModel
from pilot_models.vision_encoder.model_registry import get_vision_encoder_model
from pilot_models.policy.common.transformer import MultiLayerDecoder


class ViNT(BaseModel):
    def __init__(
            self,
            policy_model_cfg: DictConfig,
            vision_encoder_model_cfg: DictConfig,
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
        self.target_obs_enable = data_cfg.target_observation_enable
        self.target_context = data_cfg.target_context
        self.goal_condition = data_cfg.goal_condition
        # Policy model
        mha_num_attention_heads=policy_model_cfg.mha_num_attention_heads
        mha_num_attention_layers=policy_model_cfg.mha_num_attention_layers
        mha_ff_dim_factor=policy_model_cfg.mha_ff_dim_factor
        self.obs_encoding_size=policy_model_cfg.obs_encoding_size
        self.late_fusion=policy_model_cfg.late_fusion
        
        super(ViNT, self).__init__(policy_model_cfg.name,
                                context_size,
                                len_traj_pred,
                                self.learn_angle,
                                vision_encoder_model_cfg.in_channels)
        
        # Final sequence length  = context size +
        #                           current observation (1) + encoded lin observation and target (1) 
        if self.goal_condition:
            seq_len = self.context_size + 2
        else:
            seq_len = self.context_size + 1 
        
        # pred horizon # TODO: remove dupliactes in refactoring
        self.pred_horizon = self.len_trajectory_pred
        # action dom # TODO: remove duplicates. change names in refactoring
        self.action_dim = self.num_action_params
        self.action_horizon = data_cfg.action_horizon
        
        self.vision_encoder = get_vision_encoder_model(vision_encoder_model_cfg, data_cfg)
        # self.vision_encoder = replace_bn_with_gn(self.vision_encoder)
        
        # Linear input encoder #TODO: modify num_obs_features to be argument
        num_obs_features = policy_model_cfg.num_lin_features   # (now its 2)
        target_context_size = context_size if self.target_context else 0
        num_obs_features *= (target_context_size + 1)  # (context+1)
        
        if self.target_obs_enable:
            num_lin_features = num_obs_features + 2 #policy_model_cfg.num_target_features # x y
        else:
            num_lin_features = 2
        self.lin_encoding_size = policy_model_cfg.lin_encoding_size  # should match obs_encoding_size for easy concat

        # sum of features in current_rel_pos_to_target & goal_rel_pos_to_obj

        # think of a better encoding
        self.lin_encoder = nn.Sequential(nn.Linear(num_lin_features, self.lin_encoding_size // 2),
                                        nn.ReLU(),
                                        nn.Linear(self.lin_encoding_size // 2, self.lin_encoding_size))

        self.num_obs_features = self.vision_encoder.get_in_feateures()
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

        # NOT in use
        # self.dist_predictor = nn.Sequential(
        #     nn.Linear(32, 1),
        # )
        
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.action_horizon * self.action_dim),
        )

    def infer_vision_encoder(self,obs_img: torch.tensor):
        # split the observation into context based on the context size
        # Currently obs_img size is [batch_size, C*(self.context_size+1), H, W] | for example: [16, 3*(5+1), 64, 85]
        obs_img = torch.split(obs_img, self.in_channels, dim=1)

        # Currently obs_img size is [self.context_size+1, batch_size, C, H, W] | for example: [6, 16, 3, 64, 85]
        obs_img = torch.concat(obs_img, dim=0)

        # Currently obs_img size is [batch_size*(self.context_size+1), C, H, W] | for example: [96, 3, 64, 85]
        obs_encoding = self.vision_encoder(obs_img)

        # (Encoded) Currently obs_encoding size is [batch_size*(self.context_size+1), source_encoder_out_features] | for example: [96, 1280]
        obs_encoding = self.compress_obs_enc(obs_encoding)

        # (Compressed) Currently obs_encoding size is [batch_size*(self.context_size + 1), self.obs_encoding_size (= a param from config)] | for example: [96, 512]
        obs_encoding = obs_encoding.reshape((self.context_size + 1, -1, self.obs_encoding_size))
        # (reshaped) Currently obs_encoding is [self.context_size + 1, batch_size, self.obs_encoding_size], note that the order is flipped | for example: [6, 16, 512]
        obs_encoding = torch.transpose(obs_encoding, 0, 1)
        # (transposed) Currently obs_encoding size is [batch_size, self.context_size+1, self.obs_encoding_size]

        return obs_encoding
    
    def infer_linear_encoder(self, linear_input: torch.tensor):
            lin_encoding = self.lin_encoder(linear_input)
            if len(lin_encoding.shape) == 2:
                lin_encoding = lin_encoding.unsqueeze(1)
            # currently, the size of goal_encoding is [batch_size, 1, self.goal_encoding_size]
            assert lin_encoding.shape[2] == self.lin_encoding_size
            
            return lin_encoding
    
    def forward(
            self, obs_img: torch.tensor, curr_rel_pos_to_target: torch.tensor = None, goal_rel_pos_to_target: torch.tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        obs_encoding =  self.infer_vision_encoder(obs_img=obs_img)
        
        
        #TODO: in one function
        #TODO: convert to [d,cos(theta), sin(theta) rep, normalize d by max_depth[meters] and min of (0.1)
        #TODO: cat based on context size
        #TODO: do this for curr and goal as well

        if self.goal_condition:
            if self.target_obs_enable:
                linear_input = torch.cat((curr_rel_pos_to_target, goal_rel_pos_to_target), dim=1)
            else:
                # print("here")
                linear_input = torch.as_tensor(goal_rel_pos_to_target, dtype=torch.float32)

            # lin_encoding = self.lin_encoder(linear_input)
            # if len(lin_encoding.shape) == 2:
            #     lin_encoding = lin_encoding.unsqueeze(1)
            # # currently, the size of goal_encoding is [batch_size, 1, self.goal_encoding_size]
            # assert lin_encoding.shape[2] == self.lin_encoding_size
            lin_encoding = self.infer_linear_encoder(linear_input=linear_input)
        
            tokens = torch.cat((obs_encoding, lin_encoding), dim=1)  # >> Concat the lin_encoding as a token too
        
        else:       # No Goal condition >> take the obs_encoding as the tokens
            tokens = obs_encoding

        final_repr = self.decoder(tokens)

        # currently, the size is [batch_size, 32]
        action_pred_deltas = self.action_predictor(final_repr)

        # augment outputs to match labels size-wise
        action_pred_deltas = action_pred_deltas.reshape(
            (action_pred_deltas.shape[0], self.action_horizon, self.action_dim)
        )

        action_pred = torch.zeros_like(action_pred_deltas)

        action_pred[:, :, :2] = torch.cumsum(
            action_pred_deltas[:, :, :2], dim=1
        )  # convert position deltas into waypoints in local coords
        if self.learn_angle:
            action_pred[:, :, 2:] = F.normalize(
                action_pred_deltas[:, :, 2:].clone(), dim=-1
            )  # normalize the angle prediction
        return action_pred

    def _compute_losses(
        self,
            action_label: torch.Tensor,
            action_pred: torch.Tensor,
            action_mask: torch.Tensor = None,
            control_magnitude: torch.Tensor = None,
    ):
        """
        Compute losses for distance and action prediction.
        """

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

        total_loss = action_loss if control_magnitude is None else control_magnitude*action_loss
        results["total_loss"] = total_loss

        return results