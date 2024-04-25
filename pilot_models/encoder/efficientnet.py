from efficientnet_pytorch import EfficientNet as ecn
from pilot_models.encoder.base_model import BaseModel

class EfficientNet(BaseModel):
    def __init__(self, version="efficientnet-b0", in_channels=3, pretrained=False) -> None:
        super().__init__(in_channels=in_channels, pretrained=pretrained)
        
        self.version = version
        self.model = ecn.from_name(self.version, in_channels=self.in_channels)  # context

    def get_model(self):
        return self.model
    
    def get_in_feateures(self):
        return self.model._fc.in_features

    def extract_features(self, img):
        
        # get the observation encoding
        obs_encoding = self.model.extract_features(img)
        # currently the size is [batch_size*(self.context_size + 1), 1280, H/32, W/32]
        obs_encoding = self.model._avg_pooling(obs_encoding)
        # currently the size is [batch_size*(self.context_size + 1), 1280, 1, 1]
        if self.model._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.model._dropout(obs_encoding)
        # currently, the size is [batch_size, self.context_size+2, self.obs_encoding_size]
        
        return obs_encoding