import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from typing import Tuple
from omegaconf import DictConfig
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from pilot_models.policy.base_model import BaseModel
from pilot_models.model_registry import get_linear_encoder_model, get_vision_encoder_model
from pilot_utils.train.train_utils import replace_bn_with_gn
from pilot_models.policy.diffusion_policy import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from pilot_models.policy.common.transformer import MultiLayerDecoder, PositionalEncoding
from pilot_utils.utils import deltas_to_actions


class CNNMLPPolicy(BaseModel):
    def __init__(
            self,
            policy_model_cfg: DictConfig,
            vision_encoder_model_cfg: DictConfig,
            linear_encoder_model_cfg: DictConfig,
            data_cfg: DictConfig 
    ) -> None:
        """

        """
        
        ## Init BaseModel
        context_size=data_cfg.context_size
        pred_horizon=data_cfg.pred_horizon
        learn_angle=data_cfg.learn_angle
        channels=vision_encoder_model_cfg.in_channels
        super(CNNMLPPolicy, self).__init__(policy_model_cfg.name,
                                context_size,
                                pred_horizon,
                                learn_angle,
                                channels)

        # Data config
        self.target_context_enable = data_cfg.target_context_enable
        self.goal_condition = data_cfg.goal_condition

        seq_len = self.context_size + 1
        if self.goal_condition:
            seq_len+=1
            if self.target_context_enable:
                seq_len+=1

        self.action_horizon = data_cfg.action_horizon
        
        # Vision encoder for context images
        self.vision_encoder = get_vision_encoder_model(vision_encoder_model_cfg, data_cfg)
        vision_encoding_size = vision_encoder_model_cfg.vision_encoding_size
        self.vision_features_dim = self.vision_encoder.get_in_feateures()*(self.context_size+1)
        self.obs_encoding_size = vision_encoding_size ## TODO: check
        if self.vision_features_dim != vision_encoding_size:
            self.compress_obs_enc = nn.Linear(self.vision_features_dim, vision_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()
        

        lin_encoding_size = linear_encoder_model_cfg.lin_encoding_size    

        if self.target_context_enable:
            self.lin_encoder = get_linear_encoder_model(linear_encoder_model_cfg,data_cfg)
        
        # Observations encoding size
        assert vision_encoding_size == lin_encoding_size, "encoding vector of lin and vision encoders must be equal in their final dim representation"

        self.action_predictor = nn.Sequential(nn.Linear(vision_encoding_size, vision_encoding_size // 2),
                                        nn.ReLU(),
                                        nn.Linear(vision_encoding_size // 2, vision_encoding_size // 4),
                                        nn.ReLU(),
                                        nn.Linear(vision_encoding_size // 4, self.action_horizon * self.action_dim))


    def infer_vision_encoder(self,obs_img: torch.tensor):
        # split the observation into context based on the context size
        # Currently obs_img size is [batch_size, C*(self.context_size+1), H, W] | for example: [16, 3*(5+1), 64, 85]
        obs_img = torch.split(obs_img, self.in_channels, dim=1)

        # Currently obs_img size is [self.context_size+1, batch_size, C, H, W] | for example: [6, 16, 3, 64, 85]
        obs_img = torch.concat(obs_img, dim=0)

        # Currently obs_img size is [batch_size*(self.context_size+1), C, H, W] | for example: [96, 3, 64, 85]
        obs_encoding = self.vision_encoder(obs_img)

        obs_encoding = obs_encoding.reshape((-1, self.vision_features_dim))
        
        # (Encoded) Currently obs_encoding size is [batch_size*(self.context_size+1), source_encoder_out_features] | for example: [96, 1280]
        obs_encoding = self.compress_obs_enc(obs_encoding)

        # (Compressed) Currently obs_encoding size is [batch_size*(self.context_size + 1), self.obs_encoding_size (= a param from config)] | for example: [96, 512]
        obs_encoding = obs_encoding.reshape((1, -1, self.obs_encoding_size))
        
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
            fused_tensor = fused_tensor.sum(dim=1, keepdim=True)
        else:
            fused_tensor = torch.cat(modalities, dim=1)
            fused_tensor = fused_tensor.sum(dim=1, keepdim=True)

        return fused_tensor

    def infer_action(self,final_encoded_condition: torch.Tensor):
        
        # final_repr = self.decoder(tokens)

        # currently, the size is [batch_size, 32]
        action_pred_deltas = self.action_predictor(final_encoded_condition)

        # augment outputs to match labels size-wise
        action_pred_deltas = action_pred_deltas.reshape(
            (action_pred_deltas.shape[0], self.action_horizon, self.action_dim)
        )

        return action_pred_deltas

    @torch.inference_mode()
    def infer_actions(self, obs_img, curr_rel_pos_to_target, goal_rel_pos_to_target, input_goal_mask, normalized_action_context):
        # Predict the noise residual
        obs_encoding_condition = self("vision_encoder",obs_img=obs_img)

        if self.target_context_enable:
            linear_input = torch.concatenate([curr_rel_pos_to_target.flatten(1),
                                            normalized_action_context.flatten(1)], axis=1)

            lin_encoding = self("linear_encoder",
                                    curr_rel_pos_to_target=linear_input)
                            
            modalities = [obs_encoding_condition, lin_encoding]

            final_encoded_condition = self("fuse_modalities",
                                                modalities=modalities)
        else:
            final_encoded_condition = obs_encoding_condition

        cnn_mlp_output = self("action_pred",
                                final_encoded_condition=final_encoded_condition)

        # action_pred = action_pred[:,:self.action_horizon,:]
        action_pred = deltas_to_actions(deltas=cnn_mlp_output,
                                        pred_horizon=self.pred_horizon,
                                        action_horizon=self.action_horizon,
                                        learn_angle=self.learn_angle)
        
        return action_pred
    
    def forward(self, func_name, **kwargs):
        
        if func_name == "vision_encoder" :
            output = self.infer_vision_encoder(kwargs["obs_img"])
        
        elif func_name == "linear_encoder":
            output = self.infer_linear_encoder(kwargs["curr_rel_pos_to_target"])
        
        elif func_name == "fuse_modalities":
            if "mask" in kwargs:
                output = self.fuse_modalities(kwargs["modalities"],kwargs["mask"])
            else:
                output = self.fuse_modalities(kwargs["modalities"])

        elif func_name == "action_pred":
            output = self.infer_action(kwargs["final_encoded_condition"])
        else:
            raise NotImplementedError
        
        return output

    def train(self, mode: bool = True):
        super(CNNMLPPolicy, self).train(mode)
        torch.cuda.empty_cache()
        return self

    def eval(self):
        super(CNNMLPPolicy, self).eval()
        torch.cuda.empty_cache()
        return self
    
    def to(self, device):
        """
        Override the default `to` method to move mask tensors to the specified device.
        """
        super(CNNMLPPolicy, self).to(device)
        return self
