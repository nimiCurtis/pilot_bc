from pilot_models.linear_encoder.base_model import BaseModel
import torch
from torch import nn
from torch.nn.utils import spectral_norm
class MLP(BaseModel):
    """Multilayer Perceptron with last layer linear.

    Args:
        input_size (int): number of inputs
        hidden_sizes (list): can be empty list for none (linear model).
        output_size: linear layer at output, or if ``None``, the last hidden size will be the output size and will have nonlinearity applied
        nonlinearity: torch nonlinearity Module (not Functional).
    """

    def __init__(self,
            linear_encoder_config, data_config,  # Module, not Functional.
            ):
        
        super().__init__(linear_encoder_config, data_config)
        
        
        nonlinearity = getattr(nn,linear_encoder_config.nonlinearity)
        input_size = self.linear_input_dim
        output_size = self.lin_encoding_size
        hidden_sizes = linear_encoder_config.hidden_sizes
        
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
        

    def extract_features(self, input):
        """Compute the model on the input, assuming input shape [B,input_size]."""
        return self.model(input)

    @property
    def output_size(self):
        """Retuns the output size of the model."""
        return self._output_size



# class SpectralNormModel(nn.Module):
#     def __init__(self):
#         super(SpectralNormModel, self).__init__()
#         self.conv = spectral_norm(nn.Conv2d(3, 64, 3))
#         self.fc = spectral_norm(nn.Linear(64, 10))

#     def forward(self, x):
#         x = self.conv(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x


# if __name__ == "__main__":
#     model = SpectralNormModel()
#     a=1