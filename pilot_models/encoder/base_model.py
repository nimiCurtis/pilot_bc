
from torch import nn


class BaseModel(nn.Module):
    def __init__(self, late_fusion: bool = False, in_channels: int = 3, pretrained: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.pretrained = pretrained

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

    def extract_features(self, img):
        """
        Extract features from an input image or batch of images.
        """
        raise NotImplementedError("Subclasses must implement this method.")