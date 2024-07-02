import torch
import torch.nn as nn
import torch.nn.functional as F

from pilot_models.linear_encoder.base_model import BaseModel

class TCN(BaseModel):

    def __init__(self, linear_encoder_config, data_config) -> None:
        super().__init__(linear_encoder_config)

        context_size = data_config.context_size
        num_inputs = self.num_lin_features
        kernel_size = linear_encoder_config.kernel_size
        drop_out = linear_encoder_config.drop_out
        emb_size = self.lin_encoding_size
        self.tcn_observations = TemporalConvNet(num_inputs=num_inputs,
                                                num_channels=[emb_size // 4, emb_size],
                                                emb_size=emb_size,
                                                kernel_size=kernel_size,
                                                dropout=drop_out)

    def get_model(self):
        return self

    def extract_features(self, curr_rel_pos_to_target):
        linear_features = self.tcn_observations(curr_rel_pos_to_target)
        return linear_features

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, emb_size=256, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x.transpose(1, 2)).transpose(1, 2)  # Transpose to (batch, channels, seq_len) for Conv1d, then back
        return x



