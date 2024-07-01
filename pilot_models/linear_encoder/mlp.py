from pilot_models.linear_encoder.base_model import BaseModel
import torch
from torch import nn

class MLP(BaseModel):

    def __init__(self, linear_encoder_config, data_config) -> None:

        super().__init__(linear_encoder_config)

        context_size = data_config.context_size
        obs_num_lin_features = self.num_lin_features*(context_size+1)

        self.fc_observations = nn.Sequential(nn.Linear(obs_num_lin_features, self.lin_encoding_size // 8),
                                        nn.ReLU())
        self.fc_goal = nn.Sequential(nn.Linear(self.num_lin_features, self.lin_encoding_size // 8),
                                        nn.ReLU())
        self.fc_head = nn.Sequential(nn.Linear(self.lin_encoding_size // 4, self.lin_encoding_size // 2),
                                        nn.ReLU(),
                                        nn.Linear(self.lin_encoding_size // 2, self.lin_encoding_size))
    
    def get_model(self):
        return self

    def extract_features(self, curr_rel_pos_to_target, goal_rel_pos_to_target):
        curr_rel_pos_to_target = torch.flatten(curr_rel_pos_to_target,start_dim=1)
        curr_obs_encoding = self.fc_observations(curr_rel_pos_to_target)
        goal_encoding = self.fc_goal(goal_rel_pos_to_target)
        linear_input = torch.cat((curr_obs_encoding, goal_encoding), dim=1)
        linear_features = self.fc_head(linear_input)
        return linear_features

