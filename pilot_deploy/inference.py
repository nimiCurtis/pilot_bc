import os
from omegaconf import DictConfig
import torch
from torch import nn
import numpy as np
from typing import Tuple

from pilot_train.data.pilot_dataset import PilotDataset
from pilot_utils.utils import (
    get_delta,
    to_numpy,
    unnormalize_data,
    calculate_sin_cos,
    tic, toc
)
from pilot_models.policy.model_registry import get_policy_model
from pilot_config.config import get_inference_model_config, get_robot_config

# Path to checkpoints folder
CKPTH_PATH = os.path.join(os.path.dirname(__file__),
                        "checkpoints")

# TODO: need to update
class InferenceDataset(PilotDataset):
    """
    Dataset class specifically designed for inference, extending the PilotDataset.
    
    Attributes:
        data_cfg (DictConfig): Configuration related to data.
        datasets_cfg (DictConfig): Configuration for various datasets.
        robot_dataset_cfg (DictConfig): Configuration for the robot's dataset.
        dataset_name (str): The name of the dataset to be used.
        data_split_type (str): Specifies which data split to use (e.g., 'train', 'test').
    """

    def __init__(self, data_cfg: DictConfig,
                datasets_cfg: DictConfig,
                robot_dataset_cfg: DictConfig,
                dataset_name: str,
                data_split_type: str):
        """
        Initialize the InferenceDataset instance.

        Args:
            data_cfg (DictConfig): Data configuration.
            datasets_cfg (DictConfig): Dataset configuration.
            robot_dataset_cfg (DictConfig): Robot dataset configuration.
            dataset_name (str): Dataset name.
            data_split_type (str): Type of data split.
        """
        super().__init__(data_cfg, datasets_cfg, robot_dataset_cfg, dataset_name, data_split_type)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        """
        Retrieve a specific data sample by index.

        Args:
            i (int): Index of the data sample.

        Returns:
            Tuple[torch.Tensor]: A tuple containing tensors for observation image, 
                                relative positions, actions, etc.
        """
        # Access the data at index `i`
        f_curr, curr_properties, curr_time, max_goal_dist = self.index_to_data[i]
        f_goal, goal_time, goal_is_negative = self._sample_goal(f_curr, curr_time, max_goal_dist)

        # Load images for the observation context
        context = []
        if self.context_type == "temporal":
            # sample the last self.context_size times from interval [0, curr_time)
            context_times = list(
                range(
                    curr_time + -self.context_size * self.waypoint_spacing,
                    curr_time + 1,
                    self.waypoint_spacing,
                )
            )
            context = [(f_curr, t) for t in context_times]
        else:
            raise ValueError(f"Invalid context type {self.context_type}")

        obs_image = torch.cat([
            self._load_image(f, t) for f, t in context
        ])

        # Load other trajectory data
        curr_traj_data = self._get_trajectory(f_curr)
        curr_traj_len = len(curr_traj_data["position"])
        assert curr_time < curr_traj_len, f"{curr_time} and {curr_traj_len}"


        # Load current position rel to target. use f_curr and curr_time.
        curr_target_traj_data = self._get_trajectory(f_curr, target=True)
        
        # Take context of target rel pos or only the recent
        if self.target_context:
            curr_rel_pos_to_target = torch.cat([
                torch.as_tensor(curr_target_traj_data[t]["position"][:2], dtype=torch.float32) for f, t in context
            ])
        else:
            curr_rel_pos_to_target = curr_target_traj_data[curr_time]["position"][:2] # Takes the [x,y] 

        # Load goal position relative to target
        goal_target_traj_data = self._get_trajectory(f_goal, target=True)
        goal_target_traj_data_len = len(goal_target_traj_data)
        assert goal_time < goal_target_traj_data_len, f"{goal_time} an {goal_target_traj_data_len}"
        goal_rel_pos_to_target = goal_target_traj_data[goal_time]["position"][:2] # Takes the [x,y]

        # Compute actions
        action_stats = self._get_action_stats(curr_properties,self.waypoint_spacing) ## added
        actions, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time, action_stats)
        
        # Compute timesteps distances
        if goal_is_negative:
            distance = self.max_dist_cat
        else:
            distance = (goal_time - curr_time) // self.waypoint_spacing
            assert (goal_time - curr_time) % self.waypoint_spacing == 0, f"{goal_time} and {curr_time} should be separated by an integer multiple of {self.waypoint_spacing}"

        actions_torch = torch.as_tensor(actions, dtype=torch.float32)
        if self.learn_angle:
            actions_torch = calculate_sin_cos(actions_torch)

        #TODO: check and modify
        action_mask = (
            (distance < self.max_action_distance) and
            (distance > self.min_action_distance) and
            (not goal_is_negative)
        )

        # TODO: modify the return 
        return (
            # STATE : composed from context + current position of the target with relate to the camera
            torch.as_tensor(obs_image, dtype=torch.float32), # [C*(context+1),H,W]
            torch.as_tensor(curr_rel_pos_to_target, dtype=torch.float32), # change to goal_rel_pos_to_target
            # GOAL : composed only from the + "desired" position of the target with relate to the camera
            torch.as_tensor(goal_rel_pos_to_target, dtype=torch.float32), # change to goal_rel_pos_to_target
            # ACTION: composed from the requiered normalized waypoints to reach the desired goal
            actions_torch,
            # OTHER: TODO: check what is necassery
            torch.as_tensor(goal_pos, dtype=torch.float32), # goal_robot_pos_in_local_coords
            torch.as_tensor(self.dataset_index, dtype=torch.int64), 
            torch.as_tensor(action_mask, dtype=torch.float32),
        )

class PilotAgent(nn.Module):
    """
    PilotAgent class that handles policy-based predictions for pilot models.
    
    Attributes:
        model (nn.Module): The policy model to use.
        action_stats (dict): Statistical properties related to actions.
    """
    
    def __init__(self, data_cfg: DictConfig,
                policy_model_cfg: DictConfig,
                encoder_model_cfg: DictConfig,
                robot: str, wpt_i: int, frame_rate: float):
        """
        Initialize the PilotAgent instance.

        Args:
            data_cfg (DictConfig): Configuration related to data.
            policy_model_cfg (DictConfig): Configuration for policy models.
            encoder_model_cfg (DictConfig): Configuration for encoder models.
            robot (str): Robot name for configuration purposes.
            wpt_i (int): Index of the specific waypoint to retrieve.
            frame_rate (float): Sampling rate of the data
        """
        super().__init__()

        
        self.wpt_i = wpt_i
        self.frame_rate = frame_rate
        
        self.model = get_policy_model(policy_model_cfg=policy_model_cfg, encoder_model_cfg=encoder_model_cfg, data_cfg=data_cfg)
        robot_properties = get_robot_config(robot)[robot]
        self.action_stats = self._get_action_stats(robot_properties)

    def load(self, model_name):
        """
        Load a pre-trained model.

        Args:
            model_name (str): The name of the pre-trained model.
        """
        model_path = os.path.join(CKPTH_PATH, model_name, "best_model.pth")
        checkpoint = torch.load(model_path)

        state_dict = checkpoint["model_state_dict"]
        self.model.load_state_dict(state_dict, strict=False)

        self.model.eval()

    def to(self, device):
        """
        Set the device to which the model should be transferred.

        Args:
            device (torch.device): Device to use.
        """
        self.model.to(device=device)
        self.device = device

    def forward(self, obs_img: torch.Tensor, curr_rel_pos_to_target: torch.Tensor = None, goal_rel_pos_to_target: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model to predict the next waypoint.

        Args:
            obs_img (torch.Tensor): Observation image.
            curr_rel_pos_to_target (torch.Tensor): Current relative position to the target.
            goal_rel_pos_to_target (torch.Tensor): Goal relative position to the target.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predicted waypoint and corresponding action.
        """
        predicted_waypoints_normalized = self.predict_raw(obs_img, curr_rel_pos_to_target, goal_rel_pos_to_target)
        predicted_waypoints_normalized = to_numpy(predicted_waypoints_normalized)
        predicted_waypoint = self.get_waypoint(predicted_waypoints_normalized)

        return predicted_waypoint

    def get_waypoint(self, normalized_waypoints):
        """
        Calculate a waypoint given normalized waypoints.

        Args:
            normalized_waypoints (np.ndarray): Normalized waypoints.
            wpt_i (int): Index of the specific waypoint to retrieve.

        Returns:
            np.ndarray: Calculated waypoint.
        """

        cos_sin_angles = normalized_waypoints[:, 2:] # cos_sin is normalized anyway
        ndeltas = get_delta(normalized_waypoints[:, :2])
        deltas = unnormalize_data(ndeltas, self.action_stats["pos"])
        waypoints = np.cumsum(deltas, axis=0)
        waypoints = np.concatenate([waypoints, cos_sin_angles],axis=1)

        return waypoints

    def _get_action_stats(self, properties):
        """
        Retrieve the statistical properties for the actions.

        Args:
            properties (dict): Robot properties including velocity limits.

        Returns:
            dict: Action statistics for position and yaw.
        """

        frame_rate = self.frame_rate
        lin_vel_lim = properties['max_lin_vel']
        ang_vel_lim = properties['max_ang_vel']

        return {'pos': {'max': lin_vel_lim / frame_rate, 'min': -lin_vel_lim / frame_rate},
                'yaw': {'max': ang_vel_lim / frame_rate, 'min': -ang_vel_lim / frame_rate}}

    def predict_raw(self, obs_img: torch.Tensor, curr_rel_pos_to_target: torch.Tensor, goal_rel_pos_to_target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate raw predictions using the policy model.

        Args:
            obs_img (torch.Tensor): Observation image.
            curr_rel_pos_to_target (torch.Tensor): Current relative position to the target.
            goal_rel_pos_to_target (torch.Tensor): Goal relative position to the target.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Raw normalized actions predicted by the model.
        """
        
        
        context_queue = obs_img.unsqueeze(0).to(self.device)
        target_context_queue, goal_to_target = None, None
        
        if curr_rel_pos_to_target is not None:
            target_context_queue = curr_rel_pos_to_target.unsqueeze(0).to(self.device)
        
        if goal_rel_pos_to_target is not None:
            goal_to_target = goal_rel_pos_to_target.unsqueeze(0).to(self.device)

        with torch.no_grad():
            normalized_actions = self.model(context_queue, target_context_queue, goal_to_target)

        return normalized_actions[0]  # no batch dimension

def get_inference_config(model_name):
    """
    Retrieve inference configuration based on the model name.

    Args:
        model_name (str): Name of the model.

    Returns:
        Tuple: Inference model configuration.
    """
    model_source_dir = os.path.join(CKPTH_PATH, model_name)
    return get_inference_model_config(model_source_dir, rt=True)



def main():
    """
    Performing an example of inference using the PilotPlanner model.
    Loads the model, sets up the dataset, and evaluates the predictions.
    """
    # Set the name of the model to load and evaluate
    model_name = "pilot-turtle-static-follower_2024-05-02_12-38-32"

    # Retrieve the model's inference configuration
    data_cfg, datasets_cfg, policy_model_cfg, encoder_model_cfg, device = get_inference_config(model_name=model_name)

    # Define the robot name and retrieve the corresponding dataset configuration
    robot = "turtlebot"
    robot_dataset_cfg = datasets_cfg[robot]

    # Specify the type of data split to use for testing
    data_split_type = "test"
    
    # Choose the device for computation 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "cuda" else "cpu"

    # Initialize the PilotPlanner model with the appropriate configurations
    model = PilotPlanner(data_cfg=data_cfg,
                        policy_model_cfg=policy_model_cfg,
                        encoder_model_cfg=encoder_model_cfg,
                        robot=robot,
                        wpt_i=2,
                        frame_rate=6)

    # Load the pre-trained model and move to the specified device 
    model.load(model_name=model_name)
    model.to(device=device)

    # Initialize the dataset for inference with the desired configuration
    dataset = InferenceDataset(data_cfg=data_cfg,
                            datasets_cfg=datasets_cfg,
                            robot_dataset_cfg=robot_dataset_cfg,
                            dataset_name=robot,
                            data_split_type=data_split_type)

    # Initialize variables for calculating average inference time
    dt_sum = 0
    size = 2000

    # Loop through the dataset, performing inference and timing each prediction
    for i in range(0, size):
        # Retrieve relevant data for inference, including context, ground truth actions, etc.
        context_queue, target_context_queue, goal_to_target, gt_actions_normalized, _, dataset_index, _ = dataset[i]
        
        # Convert the ground truth actions into waypoints
        gt_waypoints = to_numpy(gt_actions_normalized)
        gt_waypoint = model.get_waypoint(gt_waypoints)

        # Start timing the inference process
        t = tic()
        
        # Perform inference to predict the next waypoint
        predicted_waypoint = model(context_queue, target_context_queue, goal_to_target)
        
        # Measure the elapsed time for inference
        dt = toc(t)

        # Output the predictions and their corresponding ground truth values
        print(f"infer: dataset index: {dataset_index} | sample: {i}")
        print(f"wpt predicted: {np.round(predicted_waypoint, 5)}")
        print(f"wpt gt: {np.round(gt_waypoint, 5)}")
        print(f"inference time: {dt}[sec]")
        print()

        # Skip the first iteration (as a warmup), then accumulate the inference times
        if i > 0:
            dt_sum += dt

    # Compute the average inference time, excluding the first sample (warmup)
    dt_avg = dt_sum / (size - 1)
    print(f"inference time avg: {dt_avg}[sec]")

# Entry point for the script to execute the main function
if __name__ == "__main__":
    main()
    