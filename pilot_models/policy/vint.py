from typing import Tuple
from omegaconf import DictConfig
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from pilot_models.policy.base_model import BaseModel
from pilot_models.model_registry import get_linear_encoder_model, get_vision_encoder_model
from pilot_models.policy.common.transformer import MultiLayerDecoder, PositionalEncoding


class ViNT(BaseModel):
    def __init__(
            self,
            policy_model_cfg: DictConfig,
            vision_encoder_model_cfg: DictConfig,
            linear_encoder_model_cfg: DictConfig,
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
        
        ## Init BaseModel
        context_size=data_cfg.context_size
        pred_horizon=data_cfg.pred_horizon
        learn_angle=data_cfg.learn_angle
        channels=vision_encoder_model_cfg.in_channels
        super(ViNT, self).__init__(policy_model_cfg.name,
                                context_size,
                                pred_horizon,
                                learn_angle,
                                channels)

        # Data config
        self.target_obs_enable = data_cfg.target_observation_enable
        self.target_context = data_cfg.target_context
        self.goal_condition = data_cfg.goal_condition

        # Final sequence length  = context size +
        #                           current observation (1) + encoded lin observation and target (1) 

        if self.goal_condition:
            seq_len = self.context_size + 3
        else:
            seq_len = self.context_size + 1

        self.action_horizon = data_cfg.action_horizon
        
        # Linear encoder for time series target position
        self.lin_encoder = get_linear_encoder_model(linear_encoder_model_cfg,data_cfg)

        # Vision encoder for context images
        self.vision_encoder = get_vision_encoder_model(vision_encoder_model_cfg, data_cfg)

        target_dim = data_cfg.target_dim 
        lin_encoding_size = linear_encoder_model_cfg.lin_encoding_size    
        self.goal_encoder = nn.Sequential(nn.Linear(target_dim, lin_encoding_size // 4),
                                        nn.ReLU(),
                                        nn.Linear(lin_encoding_size // 4, lin_encoding_size // 2),
                                        nn.ReLU(),
                                        nn.Linear(lin_encoding_size // 2, lin_encoding_size))

        ## TODO: add variables assignment for obe_target_enable = false -> only one target context! 

        vision_encoding_size=vision_encoder_model_cfg.vision_encoding_size
        vision_features_dim = self.vision_encoder.get_in_feateures()
        if vision_features_dim != vision_encoding_size:
            self.compress_obs_enc = nn.Linear(vision_features_dim, vision_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()

        # Observations encoding size
        assert vision_encoding_size == lin_encoding_size, "encoding vector of lin and vision encoders must be equal in their final dim representation"
        self.obs_encoding_size = vision_encoding_size

        ### Goal masking
        mha_num_attention_heads=policy_model_cfg.mha_num_attention_heads
        mha_num_attention_layers=policy_model_cfg.mha_num_attention_layers
        mha_ff_dim_factor=policy_model_cfg.mha_ff_dim_factor
        
        # Initialize positional encoding and self-attention layers
        self.positional_encoding = PositionalEncoding(self.obs_encoding_size, max_seq_len=seq_len)
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=self.obs_encoding_size, 
            nhead=mha_num_attention_heads, 
            dim_feedforward=mha_ff_dim_factor*self.obs_encoding_size, 
            activation="gelu", 
            batch_first=True, 
            norm_first=True
        )
        self.sa_encoder = nn.TransformerEncoder(self.sa_layer, num_layers=mha_num_attention_layers)

        ## TODO:
        # self.decoder = MultiLayerDecoder(
        #     embed_dim=self.obs_encoding_size,
        #     seq_len=seq_len,
        #     output_layers=[256, 128, 64, 32],
        #     nhead=mha_num_attention_heads,
        #     num_layers=mha_num_attention_layers,
        #     ff_dim_factor=mha_ff_dim_factor,
        # )

        # self.action_predictor = nn.Sequential(
        #     nn.Linear(32, self.action_horizon * self.action_dim),
        # )

        self.action_predictor = nn.Sequential(nn.Linear(self.obs_encoding_size, self.obs_encoding_size // 2),
                                        nn.ReLU(),
                                        nn.Linear(self.obs_encoding_size // 2, self.obs_encoding_size // 4),
                                        nn.ReLU(),
                                        nn.Linear(self.obs_encoding_size // 4, self.action_horizon * self.action_dim))

        # Definition of the goal mask (convention: 0 = no mask, 1 = mask)
        self.goal_mask = torch.zeros((1, seq_len), dtype=torch.bool)
        self.goal_mask[:, -1] = True # Mask out the goal 
        self.no_mask = torch.zeros((1, seq_len), dtype=torch.bool) 
        self.all_masks = torch.cat([self.no_mask, self.goal_mask], dim=0)
        self.avg_pool_mask = torch.cat([1 - self.no_mask.float(), (1 - self.goal_mask.float()) * ((seq_len)/(self.context_size + 1))], dim=0)

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
    
    def infer_linear_encoder(self, curr_rel_pos_to_target):
        lin_encoding = self.lin_encoder(curr_rel_pos_to_target)
        if len(lin_encoding.shape) == 2:
            lin_encoding = lin_encoding.unsqueeze(1)

        # currently, the size of goal_encoding is [batch_size, 1, self.goal_encoding_size]
        return lin_encoding

    
    
    def infer_goal(self,goal_rel_pos_to_target):
        goal_encoding = self.goal_encoder(goal_rel_pos_to_target)
        if len(goal_encoding.shape) == 2:
                goal_encoding = goal_encoding.unsqueeze(1)
            # currently, the size of goal_encoding is [batch_size, 1, self.goal_encoding_size]
        return goal_encoding
    
    def infer_goal_masking(self,final_encoded_condition, goal_mask):
        
        # If a goal mask is provided, mask some of the goal tokens
        if goal_mask is not None:
            no_goal_mask = goal_mask.long()
            self.all_masks = self.all_masks.to(no_goal_mask.device)
            # select from all_masks a tensor based on the no_goal_mask tensor
            src_key_padding_mask = torch.index_select(self.all_masks, 0, no_goal_mask)

        else:
            src_key_padding_mask = None
        
        # Apply positional encoding 
        if self.positional_encoding:
            final_encoded_condition = self.positional_encoding(final_encoded_condition)

        final_encoded_condition = self.sa_encoder(final_encoded_condition, src_key_padding_mask=src_key_padding_mask)

        if src_key_padding_mask is not None:
            self.avg_pool_mask = self.avg_pool_mask.to(no_goal_mask.device)
            avg_mask = torch.index_select(self.avg_pool_mask, 0, no_goal_mask).unsqueeze(-1)

            final_encoded_condition = final_encoded_condition * avg_mask
        
        final_encoded_condition = torch.mean(final_encoded_condition, dim=1)
        
        return final_encoded_condition

    def fuse_modalities(self, modalities: List[torch.Tensor], mask: torch.Tensor = None):
        # Ensure the list of tensors is not empty
        if not modalities:
            raise ValueError("The list of modalities is empty.")
        
        # Check that all tensors have the same shape
        shape = modalities[0].shape
        
        # fused_tensor = torch.zeros_like(modalities[0]) # TODO: fix this
        for tensor in modalities:
            if tensor.shape[-1] != shape[-1]:
                raise ValueError("All tensors must have the same features dim.")

        # If mask is provided, ensure it has the correct shape
        if mask is not None:
            if mask.shape != (len(modalities[0]), len(modalities)):
                raise ValueError("The mask must have the shape (batch_size, modalities_size).")

            # Apply the mask: set masked modalities to zero
            masked_modalities = []
            for i, modality in enumerate(modalities):
                masked_modality = modality * mask[:, i].unsqueeze(1).unsqueeze(2).expand_as(modality)
                masked_modalities.append(masked_modality)
        
            fused_tensor = torch.cat(masked_modalities, dim=1)  # >> Concat the lin_encoding as a token too

        else:
            fused_tensor = modalities

        return fused_tensor

    def infer_action(self,tokens: torch.Tensor):
        
        # final_repr = self.decoder(tokens)

        # currently, the size is [batch_size, 32]
        action_pred_deltas = self.action_predictor(tokens)

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


    def forward(self, func_name, **kwargs):
        
        if func_name == "vision_encoder" :
            output = self.infer_vision_encoder(kwargs["obs_img"])
        
        elif func_name == "linear_encoder":
            output = self.infer_linear_encoder(kwargs["curr_rel_pos_to_target"])
        
        elif func_name == "goal_encoder":
            output = self.infer_goal(kwargs["goal_rel_pos_to_target"])
        
        elif func_name == "fuse_modalities":
            if "mask" in kwargs:
                output = self.fuse_modalities(kwargs["modalities"],kwargs["mask"])
            else:
                output = self.fuse_modalities(kwargs["modalities"])

        elif func_name == "goal_masking":
            output = self.infer_goal_masking(kwargs["final_encoded_condition"], kwargs["goal_mask"])

        elif func_name == "action_pred":
            output = self.infer_action(kwargs["tokens"])
        else:
            raise NotImplementedError
        
        return output

    def train(self, mode: bool = True):
        super(ViNT, self).train(mode)
        torch.cuda.empty_cache()
        return self

    def eval(self):
        super(ViNT, self).eval()
        torch.cuda.empty_cache()
        return self
    
    def to(self, device):
        """
        Override the default `to` method to move mask tensors to the specified device.
        """
        super(ViNT, self).to(device)
        self.goal_mask = self.goal_mask.to(self.device)
        self.no_mask = self.no_mask.to(self.device)
        self.all_masks = self.all_masks.to(self.device)
        self.avg_pool_mask = self.avg_pool_mask.to(self.device)
        return self