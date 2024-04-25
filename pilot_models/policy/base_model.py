import torch
import torch.nn as nn
from prettytable import PrettyTable

from typing import List, Dict, Optional, Tuple


class BaseModel(nn.Module):
    def __init__(
        self,
        name: str,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
    ) -> None:
        """
        Base Model main class
        Args:
            context_size (int): how many previous observations to used for context
            len_traj_pred (int): how many waypoints to predict in the future
            learn_angle (bool): whether to predict the yaw of the robot
        """
        super(BaseModel, self).__init__()
        self.name = name
        self.context_size = context_size
        self.learn_angle = learn_angle
        self.len_trajectory_pred = len_traj_pred
        if self.learn_angle:
            self.num_action_params = 4  # last two dims are the cos and sin of the angle
        else:
            self.num_action_params = 2

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

    def forward(
        self, obs_img: torch.tensor, goal_img: torch.tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model
        Args:
            obs_img (torch.Tensor): batch of observations
            goal_img (torch.Tensor): batch of goals
        Returns:
            dist_pred (torch.Tensor): predicted distance to goal
            action_pred (torch.Tensor): predicted action
        """
        raise NotImplementedError
    
    def _compute_losses(
        self,
            dist_label: torch.Tensor,
            action_label: torch.Tensor,
            dist_pred: torch.Tensor,
            action_pred: torch.Tensor,
            action_mask: torch.Tensor = None,
    ):
        raise NotImplementedError