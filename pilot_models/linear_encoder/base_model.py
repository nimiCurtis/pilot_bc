
from torch import nn


class BaseModel(nn.Module):
    def __init__(self, linear_encoder_cfg):
        super().__init__()
        self.name = linear_encoder_cfg.name
        self.num_lin_features = linear_encoder_cfg.num_lin_features
        self.lin_encoding_size = linear_encoder_cfg.lin_encoding_size

    def get_model(self):
        """
        Retrieve the underlying model instance.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def get_in_features(self):
        """
        Retrieve the number of input features.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def extract_features(self, x):
        """
        Extract features from an input image or batch of images.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def forward(self, x):
        return self.extract_features(x)