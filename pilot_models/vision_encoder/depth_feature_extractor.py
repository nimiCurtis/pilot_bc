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
        super().__init__(in_channels=in_channels, pretrained=pretrained)
        
        self.input_dim = self.in_channels
        self.output_dim = vision_encoder_config.get("vision_out_size")
        # Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(self.input_dim, int(self.output_dim // 16), 3, stride=2, padding=1),
            nn.BatchNorm2d(int(self.output_dim // 16)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  

            nn.Conv2d(int(self.output_dim // 16), int(self.output_dim // 8), 3, stride=2, padding=1),
            nn.BatchNorm2d(int(self.output_dim // 8)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  

            nn.Conv2d(int(self.output_dim // 8), int(self.output_dim // 4), 3, stride=1, padding=1),
            nn.BatchNorm2d(int(self.output_dim // 4)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  

            nn.Conv2d(int(self.output_dim // 4), self.output_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.output_dim),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  
        )

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




class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out





