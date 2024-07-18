import os
from omegaconf import DictConfig
import torch
from torch import nn
import numpy as np
from typing import Tuple
from torchvision import transforms

import torch.nn.functional as F
from pilot_train.data.pilot_dataset import PilotDataset
from pilot_utils.utils import (
    get_delta,
    normalize_data,
    xy_to_d_cos_sin,
    unnormalize_data,
    # actions_forward_pass,
    tic, toc,
    to_numpy,
    # get_goal_mask_tensor,
    get_action_stats,
    clip_angle
)

from pilot_utils.transforms import transform_images, ObservationTransform

from pilot_models.model_registry import get_policy_model
from pilot_config.config import get_inference_model_config, get_robot_config

from pilot_models.policy.pidiff import DiffuserScheduler

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

    def __init__(
            self,
            data_cfg: DictConfig,
            datasets_cfg: DictConfig,
            robot_dataset_cfg: DictConfig,
            dataset_name: str,
            data_split_type: str,
            transform: transforms = None
                ):
        """
        Initialize the InferenceDataset instance.

        Args:
            data_cfg (DictConfig): Data configuration.
            datasets_cfg (DictConfig): Dataset configuration.
            robot_dataset_cfg (DictConfig): Robot dataset configuration.
            dataset_name (str): Dataset name.
            data_split_type (str): Type of data split.
        """
        super().__init__(data_cfg,
                        datasets_cfg,
                        robot_dataset_cfg,
                        dataset_name,
                        data_split_type,
                        transform)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        """
        Retrieves the i-th sample from the dataset.

        Args:
            i (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the context, observation, goal, transformed context, transformed observation, transformed goal, distance label, and action label.
        """

        # Retrieve the current trajectory name, properties, current time, and max goal distance from the index
        f_curr, curr_properties, curr_time, max_goal_dist = self.index_to_data[i]

        # Sample a goal from the current trajectory or a different trajectory
        f_goal, goal_time, goal_is_negative = self._sample_goal(f_curr, curr_time, max_goal_dist)

        # Initialize the context list
        context = []
        if self.context_type == "temporal":
            # Generate a list of context times based on the current time and context size
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

        # Load images for each context time step
        obs_image = [self._load_image(f, t) for f, t in context]

        # Apply transformations to the context images
        obs_image = transform_images(obs_image, self.transform)

        # Load the current trajectory data
        curr_traj_data = self._get_trajectory(f_curr)
        curr_traj_len = len(curr_traj_data["position"])
        assert curr_time < curr_traj_len, f"{curr_time} >= {curr_traj_len}"

        # Compute the actions and normalized goal position
        action_stats = get_action_stats(curr_properties, self.waypoint_spacing)
        normalized_actions, normalized_goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time, action_stats)

        if self.goal_condition:
            # Load the current and goal target trajectory data
            curr_target_traj_data = self._get_trajectory(f_curr, target=True)
            goal_target_traj_data = self._get_trajectory(f_goal, target=True)
            goal_target_traj_data_len = len(goal_target_traj_data)
            assert goal_time < goal_target_traj_data_len, f"{goal_time} and {goal_target_traj_data_len}"

            # Get the goal position relative to the target
            goal_rel_pos_to_target = np.array(goal_target_traj_data[goal_time]["position"][:2])
            if np.any(goal_rel_pos_to_target != np.zeros_like(goal_rel_pos_to_target)):
                goal_rel_pos_to_target = xy_to_d_cos_sin(goal_rel_pos_to_target)
                goal_rel_pos_to_target[0] = normalize_data(data=goal_rel_pos_to_target[0], stats={'min': 0.1, 'max': self.max_depth / 1000})
            else:
                goal_rel_pos_to_target = np.zeros((3,))

            if self.target_context:
                # Get the context of target positions relative to the current trajectory
                np_curr_rel_pos_to_target = np.array([
                    curr_target_traj_data[t]["position"][:2] for f, t in context
                ])
            else:
                # For now, this is not in use
                np_curr_rel_pos_to_target = np.array(curr_target_traj_data[curr_time]["position"][:2])

            mask = np.sum(np_curr_rel_pos_to_target == np.zeros((2,)), axis=1) == 2
            np_curr_rel_pos_in_d_theta = np.zeros((np_curr_rel_pos_to_target.shape[0], 3))
            np_curr_rel_pos_in_d_theta[~mask] = xy_to_d_cos_sin(np_curr_rel_pos_to_target[~mask])
            np_curr_rel_pos_in_d_theta[~mask, 0] = normalize_data(data=np_curr_rel_pos_in_d_theta[~mask, 0], stats={'min': 0.1, 'max': self.max_depth / 1000})

            # Convert the context of relative positions to target into a tensor
            curr_rel_pos_to_target = torch.as_tensor(np_curr_rel_pos_in_d_theta)
        else:
            # Not in use
            curr_rel_pos_to_target = np.zeros_like((normalized_actions.shape[0], 3, 0))
            goal_rel_pos_to_target = np.array([0, 0, 0])

        # Compute the timestep distances
        if goal_is_negative:
            distance = self.max_dist_cat
        else:
            distance = (goal_time - curr_time) // self.waypoint_spacing
            assert (goal_time - curr_time) % self.waypoint_spacing == 0, f"{goal_time} and {curr_time} should be separated by an integer multiple of {self.waypoint_spacing}, target_traj_len = {goal_target_traj_data_len}"

        # Determine if the action should be masked
        action_mask = (
            (distance < self.max_action_distance) and
            (distance > self.min_action_distance) and
            (not goal_is_negative)
        )

        # Return the context, observation, goal, normalized actions, and other necessary information as tensors
        return (
            torch.as_tensor(obs_image, dtype=torch.float32),  # [C*(context+1),H,W]
            torch.as_tensor(curr_rel_pos_to_target, dtype=torch.float32),
            torch.as_tensor(goal_rel_pos_to_target, dtype=torch.float32),
            torch.as_tensor(normalized_actions, dtype=torch.float32),
            torch.as_tensor(normalized_goal_pos, dtype=torch.float32),
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
    # TASK_STATE = {"goal_directed": 1,
    #             "explore": 0}
    
    
    def __init__(self, data_cfg: DictConfig,
                policy_model_cfg: DictConfig,
                vision_encoder_cfg: DictConfig,
                linear_encoder_cfg: DictConfig,
                robot: str, wpt_i: int, frame_rate: float, waypoint_spacing: int = 2):
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
        self.model = get_policy_model(policy_model_cfg=policy_model_cfg,
                                    vision_encoder_model_cfg=vision_encoder_cfg,
                                    linear_encoder_model_cfg=linear_encoder_cfg,
                                    data_cfg=data_cfg)
        
        self.learn_angle = data_cfg.learn_angle
        self.pred_horizon = data_cfg.pred_horizon
        self.action_dim = 4 if self.learn_angle else 2
        self.action_horizon = data_cfg.action_horizon
        # For now we use only pidiff model 
        noise_scheduler_config = self.model.module.get_scheduler_config() if hasattr(self.model, "module") else self.model.get_scheduler_config()
        self.noise_scheduler = DiffuserScheduler(noise_scheduler_config)
        
        robot_properties = get_robot_config(robot)[robot]
        robot_properties.update({'frame_rate':self.frame_rate})
        self.action_stats = get_action_stats(properties=robot_properties, waypoint_spacing=waypoint_spacing)

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
        self.noise_scheduler.eval()

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
        target_context_queue, goal_to_target, goal_mask = None, None, None
        
        if curr_rel_pos_to_target is not None:
            # print(curr_rel_pos_to_target)
            # goal_mask = get_goal_mask_tensor(curr_rel_pos_to_target).to(self.device)
            target_context_queue = curr_rel_pos_to_target.unsqueeze(0).to(self.device)
            goal_mask = torch.sum((torch.sum(curr_rel_pos_to_target==torch.zeros_like(curr_rel_pos_to_target),axis=1) == curr_rel_pos_to_target.shape[0])).long()
            
            # print(goal_mask)
            
        if goal_rel_pos_to_target is not None:
            # print(goal_rel_pos_to_target)
            goal_to_target = goal_rel_pos_to_target.unsqueeze(0).to(self.device)

        with torch.no_grad():
            normalized_actions = self.infer_actions(
                obs_img=context_queue,
                curr_rel_pos_to_target=target_context_queue,
                goal_rel_pos_to_target=goal_to_target,
                input_goal_mask=goal_mask
            )

        return normalized_actions[0]  # no batch dimension
    
    def infer_actions(self, obs_img, curr_rel_pos_to_target,goal_rel_pos_to_target, input_goal_mask):
        
        obs_encoding = self.model("vision_encoder",
                                obs_img=obs_img)
        
        # Get the input goal mask 
        if input_goal_mask is not None:
            goal_mask = input_goal_mask.to(self.device)

        # TODO: add if else condition on goal_condition
        
        lin_encoding = self.model("linear_encoder",
                                curr_rel_pos_to_target=curr_rel_pos_to_target)

        modalities = [obs_encoding, lin_encoding]
        fused_modalities_encoding = self.model("fuse_modalities", modalities=modalities)
        
        goal_encoding = self.model("goal_encoder",
                                goal_rel_pos_to_target=goal_rel_pos_to_target)
        
        final_encoded_condition = torch.cat((fused_modalities_encoding, goal_encoding), dim=1)  # >> Concat the lin_encoding as a token too

        final_encoded_condition = self.model("goal_masking",final_encoded_condition=final_encoded_condition, goal_mask=goal_mask)

        # else:       # No Goal condition >> take the obs_encoding as the tokens # not in use!!!
        #     final_encoded_condition = obs_encoding

        # initialize action from Gaussian noise
        noisy_diffusion_output = torch.randn(
            (len(final_encoded_condition), self.pred_horizon, self.action_dim),device=self.device)
        diffusion_output = noisy_diffusion_output
        
        for k in self.noise_scheduler.timesteps():
            # predict noise
            noise_pred = self.model("noise_pred",
                                        noisy_action= diffusion_output,
                                        timesteps = k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(self.device),
                                        final_encoded_condition = final_encoded_condition)

            # inverse diffusion step (remove noise)
            diffusion_output = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=diffusion_output
            )
        
        # diffusion output should be denoised action deltas
        action_pred_deltas = diffusion_output
        
        # augment outputs to match labels size-wise
        action_pred_deltas = action_pred_deltas.reshape(
            (action_pred_deltas.shape[0], self.pred_horizon, self.action_dim)
        )

        # Init action traj
        action_pred = torch.zeros_like(action_pred_deltas)
        
        ## Cumsum 
        action_pred[:, :, :2] = torch.cumsum(
            action_pred_deltas[:, :, :2], dim=1
        )  # convert position and orientation deltas into waypoints in local coords

        if self.model.learn_angle:
            action_pred[:, :, 2:] = F.normalize(
                action_pred_deltas[:, :, 2:].clone(), dim=-1
            )  # normalize the angle prediction to be fit with orientation representation [cos(theta), sin(theta)] >> (-1,1) normalization
            
        action = action_pred[:,:self.action_horizon,:]
        
        return action

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

def position_error(predicted: np.ndarray, target: np.ndarray) -> float:
    """
    Calculate the Mean Squared Error (MSE) loss between two numpy arrays.

    Args:
        predicted (np.ndarray): Predicted values.
        target (np.ndarray): Ground truth values.

    Returns:
        float: MSE loss.
    """
    return np.mean(np.linalg.norm(predicted-target,axis=1))

def angle_error(predicted: np.ndarray, target: np.ndarray) -> float:
    """
    Calculate the average angular error between predicted and target angles.

    Args:
        predicted (np.ndarray): Predicted angle values.
        target (np.ndarray): Ground truth angle values.

    Returns:
        float: The mean angular error.
    """
    
    error_arr = [clip_angle(predicted[i]) - clip_angle(target[i]) for i in range(len(predicted))]
    
    return np.mean(error_arr)

def main():
    """
    Performing an example of inference using the PilotPlanner model.
    Loads the model, sets up the dataset, and evaluates the predictions.
    """
    # Set the name of the model to load and evaluate
    model_name = "pilot-target-tracking_2024-07-14_17-47-20"

    # Retrieve the model's inference configuration
    data_cfg, datasets_cfg, policy_model_cfg, vision_encoder_cfg, linear_encoder_cfg, device = get_inference_config(model_name=model_name)
    # Define the robot name and retrieve the corresponding dataset configuration
    robot = "nimrod"
    robot_dataset_cfg = datasets_cfg[robot]

    # Specify the type of data split to use for testing
    data_split_type = "test"
    
    # Choose the device for computation 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "cuda" else "cpu"

    # Initialize the PilotPlanner model with the appropriate configurations
    wpt_i = 2
    frame_rate = 7
    model = PilotAgent(data_cfg=data_cfg,
                                policy_model_cfg=policy_model_cfg,
                                vision_encoder_cfg=vision_encoder_cfg,
                                linear_encoder_cfg=linear_encoder_cfg,
                                robot=robot,
                                wpt_i=wpt_i,
                                frame_rate=frame_rate)

    # Load the pre-trained model and move to the specified device 
    model.load(model_name=model_name)
    model.to(device=device)

    # Initialize transform for test
    transform = ObservationTransform(data_cfg=data_cfg).get_transform("test")
    
    # Initialize the dataset for inference with the desired configuration
    dataset = InferenceDataset(data_cfg=data_cfg,
                            datasets_cfg=datasets_cfg,
                            robot_dataset_cfg=robot_dataset_cfg,
                            dataset_name=robot,
                            data_split_type=data_split_type,
                            transform=transform)

    # Initialize variables for calculating average inference time
    dt_sum = 0
    size = 2000
    size = min(size,len(dataset))
    action_horizon = data_cfg.action_horizon
    pos_tot_err = 0
    yaw_tot_err = 0
    
    # Loop through the dataset, performing inference and timing each prediction
    for i in range(0, size):
        # Retrieve relevant data for inference, including context, ground truth actions, etc.
        context_queue, target_context_queue, goal_to_target, gt_actions_normalized, _, dataset_index, _ = dataset[i]
        
        # Convert the ground truth actions into waypoints
        gt_waypoints = to_numpy(gt_actions_normalized)
        gt_waypoints = model.get_waypoint(gt_waypoints)
        gt_waypoints = gt_waypoints[:action_horizon,:]
        
        # Start timing the inference process
        t = tic()
        
        # Perform inference to predict the next waypoint
        predicted_waypoint = model(context_queue, target_context_queue, goal_to_target)
        
        # Measure the elapsed time for inference
        dt = toc(t)

        pos_error = position_error(predicted=predicted_waypoint[:,:2],target=gt_waypoints[:,:2])
        pos_tot_err+=pos_error
        
        hx_gt, hy_gt = gt_waypoints[:,2], gt_waypoints[:,3]
        hx_predicted, hy_predicted = predicted_waypoint[:,2], predicted_waypoint[:,3]
        
        yaw_gt = np.arctan2(hy_gt, hx_gt)
        yaw_predicted = np.arctan2(hy_predicted, hx_predicted)
        
        yaw_error = angle_error(predicted=yaw_predicted,target=yaw_gt)
        yaw_tot_err+=yaw_error
        # Output the predictions and their corresponding ground truth values
        print(f"infer: dataset index: {dataset_index} | sample: {i}")
        print(f"wpt predicted: {np.round(predicted_waypoint[wpt_i], 5)}")
        print(f"wpt gt: {np.round(gt_waypoints[wpt_i], 5)}")
        print(f"inference time: {dt}[sec]")
        print(f"Position Error: {pos_error}")
        print(f"Yaw Error: {yaw_error} [rad] | {np.rad2deg(yaw_error)} [deg]")
        print()

        # Skip the first iteration (as a warmup), then accumulate the inference times
        if i > 0:
            dt_sum += dt

    # Compute the average inference time, excluding the first sample (warmup)
    dt_avg = dt_sum / (size - 1)
    pos_avg_err = pos_tot_err/size
    yaw_avg_err = yaw_tot_err/size
    
    print(f"inference time avg: {dt_avg}[sec]")
    print(f"Position average MSE Error: {pos_avg_err}")
    print(f"Yaw average Error: {yaw_avg_err} [rad] | {np.rad2deg(yaw_avg_err)} [deg]")

# Entry point for the script to execute the main function
if __name__ == "__main__":
    main()
    