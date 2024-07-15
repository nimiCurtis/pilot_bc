from pilot_models.linear_encoder.base_model import BaseModel
import torch
from torch import nn

class MLP(BaseModel):

    def __init__(self, linear_encoder_config, data_config) -> None:

        super().__init__(linear_encoder_config)

        context_size = data_config.context_size
        obs_num_lin_features = self.num_lin_features*(context_size+1)

        self.fc = nn.Sequential(nn.Linear(obs_num_lin_features, self.lin_encoding_size // 4),
                                        nn.ReLU(),
                                        nn.Linear(self.lin_encoding_size // 4, self.lin_encoding_size // 2),
                                        nn.ReLU(),
                                        nn.Linear(self.lin_encoding_size // 2, self.lin_encoding_size))
        
    def get_model(self):
        return self

    def extract_features(self, curr_rel_pos_to_target):
        curr_rel_pos_to_target = torch.flatten(curr_rel_pos_to_target,start_dim=1)
        linear_features = self.fc(curr_rel_pos_to_target)
        # goal_encoding = self.fc_goal(goal_rel_pos_to_target)
        # linear_input = torch.cat((curr_obs_encoding, goal_encoding), dim=1)
        # linear_features = self.fc_head(linear_input)
        return linear_features

class MlpModel(torch.nn.Module):
    """Multilayer Perceptron with last layer linear.

    Args:
        input_size (int): number of inputs
        hidden_sizes (list): can be empty list for none (linear model).
        output_size: linear layer at output, or if ``None``, the last hidden size will be the output size and will have nonlinearity applied
        nonlinearity: torch nonlinearity Module (not Functional).
    """

    def __init__(
            self,
            input_size,
            hidden_sizes,  # Can be empty list or None for none.
            output_size=None,  # if None, last layer has nonlinearity applied.
            nonlinearity=torch.nn.ReLU,  # Module, not Functional.
            ):
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        elif hidden_sizes is None:
            hidden_sizes = []
        hidden_layers = [torch.nn.Linear(n_in, n_out) for n_in, n_out in
            zip([input_size] + hidden_sizes[:-1], hidden_sizes)]
        sequence = list()
        for layer in hidden_layers:
            sequence.extend([layer, nonlinearity()])
        if output_size is not None:
            last_size = hidden_sizes[-1] if hidden_sizes else input_size
            sequence.append(torch.nn.Linear(last_size, output_size))
        self.model = torch.nn.Sequential(*sequence)
        self._output_size = (hidden_sizes[-1] if output_size is None
            else output_size)

    def forward(self, input):
        """Compute the model on the input, assuming input shape [B,input_size]."""
        return self.model(input)

    @property
    def output_size(self):
        """Retuns the output size of the model."""
        return self._output_size

# Test script for MlpModel

# # Define the model parameters
# input_size = 10
# hidden_sizes = [20, 30]
# output_size = 5

# # Initialize the model
# model = MlpModel(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)

# # Print the model architecture
# print("Model architecture:\n", model)

# # Create a random input tensor with batch size 8 and the specified input size
# batch_size = 8
# random_input = torch.randn(batch_size, input_size)

# # Perform inference
# output = model(random_input)

# # Print the input and output shapes to verify
# print("Input shape:", random_input.shape)
# print("Output shape:", output.shape)

# # Print the output
# print("Output:\n", output)