import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from pilot_models.vision_encoder.base_model import BaseModel
from torch.nn.utils import spectral_norm

class DepthFeatureExtractor(BaseModel):
    def __init__(self, vision_encoder_config, data_config):
        
        in_channels = vision_encoder_config.get("in_channels")
        
        pretrained = vision_encoder_config.get("pretrained")
        
        output_dim = vision_encoder_config.get("vision_encoding_size")*2
        
        super().__init__(in_channels=in_channels, pretrained=pretrained)
        
        self.input_dim = self.in_channels
        self.output_dim = output_dim
        # Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(self.input_dim, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            nn.Conv2d(128, self.output_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.output_dim),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
        )

        # Initialize weights
        self._initialize_weights()

    def extract_features(self, img):
        obs_encoding = self.features(img)
        obs_encoding = obs_encoding.flatten(start_dim=1)
        return obs_encoding
    
    def get_in_feateures(self):
        return self.output_dim
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                init.constant_(m.bias, 0)

