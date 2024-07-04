import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from pilot_models.vision_encoder.base_model import BaseModel

class DepthFeatureExtractor(BaseModel):
    def __init__(self, vision_encoder_config, data_config):
        
        in_channels = vision_encoder_config.get("in_channels")
        pretrained = vision_encoder_config.get("pretrained")
        super().__init__(in_channels=in_channels, pretrained=pretrained)
        
        self.input_dim = self.in_channels
        self.output_dim = 1024
        # Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(self.input_dim, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 80x80

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 20x20

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 10x10

            nn.Conv2d(128, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 5x5
        )

        # Flattening the tensor and passing through Fully Connected layers
        # self.fc = nn.Sequential(
        #     nn.Linear(128 * 5 * 5, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, self.output_dim)
        # )

        # Initialize weights
        self._initialize_weights()

    # def forward(self, x):
    #     x = self.extract_features(x)
    #     # x = x.view(x.size(0), -1)  # Flatten the tensor
    #     # x = self.fc(x)
    #     return x
    
    def extract_features(self, img):
        
        # get the observation encoding
        obs_encoding = self.features(img)
        # currently the size is [batch_size*(self.context_size + 1), 1280, H/32, W/32]
        # obs_encoding = self.model._avg_pooling(obs_encoding)
        # currently the size is [batch_size*(self.context_size + 1), 1280, 1, 1]
        # if self.model._global_params.include_top:
        obs_encoding = obs_encoding.flatten(start_dim=1)
            # obs_encoding = self.model._dropout(obs_encoding)
        # currently, the size is [batch_size, self.context_size+2, self.obs_encoding_size]
        
        return obs_encoding
    
    def get_in_feateures(self):
        return 512
    
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

