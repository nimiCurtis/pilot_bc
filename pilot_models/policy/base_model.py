import torch
import torch.nn as nn
from prettytable import PrettyTable
from typing import Optional

class BaseModel(nn.Module):
    def __init__(
        self,
        name: str,
        context_size: int = 5,
        pred_horizon: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        in_channels: int = 1,
    ) -> None:
        """
        Base Model main class
        Args:
            context_size (int): how many previous observations to used for context
            pred_horizon (int): how many waypoints to predict in the future
            learn_angle (bool): whether to predict the yaw of the robot
        """
        super(BaseModel, self).__init__()
        self.name = name
        self.context_size = context_size
        self.learn_angle = learn_angle
        self.pred_horizon = pred_horizon
        if self.learn_angle:
            self.action_dim = 4  # last two dims are the cos and sin of the angle
        else:
            self.action_dim = 2
        self.in_channels = in_channels

    def flatten(self, z: torch.Tensor) -> torch.Tensor:
        z = nn.functional.adaptive_avg_pool2d(z, (1, 1))
        z = torch.flatten(z, 1)
        return z
    
    def count_parameters(self):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
        # print(table)
        print(f"Total Trainable Params: {total_params / 1e6:.2f}M")
        return total_params
    
    def to(self, device):
            self.device = device  # Update the device attribute
            return super(BaseModel, self).to(device)

    def forward(
        self, func_name, **kwargs
    ):
        """

        """
        raise NotImplementedError
    
