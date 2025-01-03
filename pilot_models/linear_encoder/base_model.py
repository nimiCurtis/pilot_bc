
from torch import nn


class BaseModel(nn.Module):
    def __init__(self, linear_encoder_cfg, data_cfg):
        super().__init__()
        self.name = linear_encoder_cfg.name
        self.lin_encoding_size = linear_encoder_cfg.lin_encoding_size
        context_size = data_cfg.context_size
        action_context_size = data_cfg.action_context_size
        action_dim = 4 if data_cfg.learn_angle else 2
        self.target_dim = data_cfg.target_dim

        self.linear_input_dim = self.target_dim


    def get_model(self):
        """
        Retrieve the underlying model instance.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def extract_features(self, x):
        """
        Extract features from an input image or batch of images.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def forward(self, curr_rel_pos_to_target):
        lin_encoding = self.extract_features(curr_rel_pos_to_target)
        assert lin_encoding.shape[-1] == self.lin_encoding_size
        return lin_encoding