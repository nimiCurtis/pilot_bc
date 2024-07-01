from pilot_models.linear_encoder.base_model import BaseModel
from torch import nn

class MLP(BaseModel):

    def __init__(self, linear_encoder_config, data_config) -> None:

        super().__init__(linear_encoder_config)
        
        context_size = data_config.context_size
        num_lin_features = self.num_lin_features*(context_size+2)
        self.model = nn.Sequential(nn.Linear(num_lin_features, self.lin_encoding_size // 2),
                                        nn.ReLU(),
                                        nn.Linear(self.lin_encoding_size // 2, self.lin_encoding_size))

    def get_model(self):
        return self.model

    def get_in_feateures(self):
        return #TODO

    def extract_features(self, x):
        x = self.model(x)
        return x