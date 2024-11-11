import os
from omegaconf import DictConfig
import torch
import json
from torch import nn
import numpy as np
from typing import Tuple
from torchvision import transforms

import torch.nn.functional as F
import torchvision.transforms.functional as TF

from pilot_models.policy.base_model import BaseModel
from pilot_train.data.pilot_dataset import PilotDataset
from pilot_utils.data.data_utils import (
    img_path_to_data,
    get_data_path,
    to_local_coords,
)

from pilot_utils.utils import (
    get_delta,
    normalize_data,
    xy_to_d_cos_sin,
    unnormalize_data,
    actions_forward_pass,
    tic, toc,
    to_numpy,
    # get_goal_mask_tensor,
    get_action_stats,
    clip_angles,
)

from pilot_utils.transforms import transform_images, ObservationTransform
from pilot_utils.visualizing import Visualizer, VIZ_IMAGE_SIZE

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
        f_curr, curr_data_properties, curr_time, max_goal_dist, min_goal_dist = self.index_to_data[i]

                # Sample a goal from the current trajectory or a different trajectory
        f_goal, goal_time, goal_is_negative = self._sample_goal(f_curr, curr_time, max_goal_dist,min_goal_dist)

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
        vision_obs_context = [self._load_image(f, t) for f, t in context]

        # Apply transformations to the context images
        vision_obs_context_tensor = transform_images(vision_obs_context, self.transform)
        
        # Load the current trajectory data
        curr_traj_data = self._get_trajectory(f_curr)
        curr_traj_len = len(curr_traj_data["position"])
        assert curr_time < curr_traj_len, f"{curr_time} >= {curr_traj_len}"

        # Compute the actions and normalized goal position
        # was self.waypoint_spacing
        action_stats = get_action_stats(curr_data_properties, self.waypoint_spacing_action)
        context_action_stats = get_action_stats(curr_data_properties, self.waypoint_spacing)

        # Stay with normalized actions (Deltas)
        normalized_actions, unormalized_prev_actions,normalized_prev_action, normalized_goal_pos, unormalized_goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time, action_stats, context_action_stats)

        # Load the current and goal target trajectory data
        target_traj_data_context = self._get_trajectory(f_curr, target=True) #fcur
        target_traj_data_goal = self._get_trajectory(f_goal, target=True) #fgoal
        target_traj_data_goal_len = len(target_traj_data_goal)
        assert goal_time < target_traj_data_goal_len, f"{goal_time} and {target_traj_data_goal_len}"

        last_det_time = self._find_last_det_time(target_traj_data_context, curr_time, self.waypoint_spacing,curr_data_properties["frame_rate"], context_times=context_times)
        vision_obs_memory = self._load_image(f_curr, last_det_time)
        vision_obs_memory_tensor = transform_images(vision_obs_memory, self.memory_transform)
        
        # Compute the relative position to target 
        rel_pos_to_target_context, goal_rel_pos_to_target, goal_pos_to_target_mask, last_det = self._compute_rel_pos_to_target(target_traj_data_context,
                                                                                                                    target_traj_data_goal,
                                                                                                                    goal_time,
                                                                                                                    curr_time,
                                                                                                                    last_det_time,
                                                                                                                    context)


        # Compute the timestep distances
        if goal_is_negative:
            distance = self.max_dist_cat
        else:
            distance = (goal_time - curr_time) // self.waypoint_spacing_action
            assert (goal_time - curr_time) % self.waypoint_spacing_action == 0, f"{goal_time} and {curr_time} should be separated by an integer multiple of {self.waypoint_spacing}, target_traj_len = {goal_target_traj_data_len}"

        # Determine if the action should be masked
        action_mask = (
            (distance <= self.max_action_distance) and
            (distance >= self.min_action_distance) and
            (not goal_is_negative)
        )

        # Return the context, observation, goal, normalized actions, and other necessary information as tensors
        return (
            vision_obs_context_tensor,  
            torch.as_tensor(rel_pos_to_target_context, dtype=torch.float32),
            torch.as_tensor(goal_rel_pos_to_target, dtype=torch.float32),
            torch.as_tensor(normalized_actions, dtype=torch.float32),
            torch.as_tensor(unormalized_prev_actions, dtype=torch.float32),
            torch.as_tensor(normalized_prev_action, dtype=torch.float32),
            torch.as_tensor(normalized_goal_pos, dtype=torch.float32),
            torch.as_tensor(unormalized_goal_pos, dtype=torch.float32),
            torch.as_tensor(self.dataset_index, dtype=torch.int64),
            torch.as_tensor(action_mask, dtype=torch.float32),
            vision_obs_memory_tensor,
            torch.as_tensor(last_det, dtype=torch.float32),
        )

    def _compute_actions(self, traj_data, curr_time, goal_time, action_stats, context_action_stats):
            """
            Computes the actions required to reach the goal from the current trajectory data.

            Args:
                traj_data (dict): Trajectory data.
                curr_time (int): Current time step.
                goal_time (int): Goal time step.
                action_stats (dict): Action statistics for normalization.

            Returns:
                tuple: Normalized actions and normalized goal position.
            """
            
            # Define the start and end indices for slicing the trajectory data
            start_index = curr_time
            end_index = curr_time + self.pred_horizon * self.waypoint_spacing_action + 1
            
            
            # Define the start and end indices for slicing the actions history data
            start_index_prev = curr_time + -self.action_context_size * self.waypoint_spacing
            end_index_prev = curr_time + self.waypoint_spacing
            
            # Extract yaw and position data from the trajectory
            yaw = np.array(traj_data["yaw"][start_index:end_index:self.waypoint_spacing_action])
            positions = np.array(traj_data["position"][start_index:end_index:self.waypoint_spacing_action])

            prev_yaw = np.array(traj_data["yaw"][start_index_prev: end_index_prev: self.waypoint_spacing])
            prev_positions =  np.array(traj_data["position"][start_index_prev: end_index_prev: self.waypoint_spacing])

            # prev_action_delta = np.concatenate([positions[0] - prev_position, np.array([yaw[0] - prev_yaw])])
            
            # Get the goal position, ensuring it does not exceed the length of the trajectory
            goal_pos = np.array(traj_data["position"][start_index:min(goal_time, len(traj_data["position"]) - 1):self.waypoint_spacing_action])

            # Handle the case where yaw has an extra dimension
            if len(yaw.shape) == 2:
                yaw = yaw.squeeze(1)

            # Ensure yaw and position arrays have the correct shape by padding if necessary
            if yaw.shape != (self.pred_horizon + 1,):
                const_len = self.pred_horizon + 1 - yaw.shape[0]
                yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
                positions = np.concatenate([positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0)

            # Assertions to ensure the shapes of yaw and positions are correct
            assert yaw.shape == (self.pred_horizon + 1,), f"{yaw.shape} and {(self.pred_horizon + 1,)} should be equal"
            assert positions.shape == (self.pred_horizon + 1, 2), f"{positions.shape} and {(self.pred_horizon + 1, 2)} should be equal"

            # Convert positions and goal to local coordinates
            waypoints = to_local_coords(positions, positions[0], yaw[0])
            goal_in_local = to_local_coords(goal_pos, positions[0], yaw[0])
            unormalized_goal_pos = goal_in_local[-1]
            
            prev_waypoints = to_local_coords(prev_positions, prev_positions[0], prev_yaw[0])
            # Ensure waypoints have the correct shape
            assert waypoints.shape == (self.pred_horizon + 1, 2), f"{waypoints.shape} and {(self.pred_horizon + 1, 2)} should be equal"

            if self.learn_angle:
                # Compute relative yaw changes and concatenate with waypoints
                yaw = yaw[1:] - yaw[0]  # yaw is relative to the initial yaw
                actions = np.concatenate([waypoints[1:], yaw[:, None]], axis=-1)
                
                prev_yaw = prev_yaw[1:] - prev_yaw[0]  # yaw is relative to the initial yaw
                prev_actions = np.concatenate([prev_waypoints[1:], prev_yaw[:, None]], axis=-1)
            else:
                actions = waypoints[1:]
                prev_actions = prev_waypoints[1:]
                
            # Always for now
            if self.normalize:
                # Normalize the actions based on provided action statistics
                normalized_actions = actions_forward_pass(actions, action_stats, self.learn_angle, norm_type=self.norm_type)
                normalized_prev_action = actions_forward_pass(prev_actions,context_action_stats,self.learn_angle, norm_type=self.norm_type)
                # Normalize the goal position in local coordinates
                # normalized_goal_pos = normalize_data(goal_in_local, action_stats['pos'])
                normalized_goal_pos = actions_forward_pass(goal_in_local, action_stats, learn_angle=False, norm_type=self.norm_type)
                normalized_goal_pos = normalized_goal_pos[-1]
            # Assertion to ensure the shape of normalized actions is correct
            assert normalized_actions.shape == (self.pred_horizon, self.num_action_params), f"{normalized_actions.shape} and {(self.pred_horizon, self.num_action_params)} should be equal"

            return normalized_actions,prev_actions,normalized_prev_action, normalized_goal_pos, unormalized_goal_pos

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
                robot: str, wpt_i: int, frame_rate: float, waypoint_spacing: int = 1):
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
        self.model: BaseModel  = get_policy_model(policy_model_cfg=policy_model_cfg,
                                    vision_encoder_model_cfg=vision_encoder_cfg,
                                    linear_encoder_model_cfg=linear_encoder_cfg,
                                    data_cfg=data_cfg)
        
        self.learn_angle = data_cfg.learn_angle
        self.pred_horizon = data_cfg.pred_horizon
        self.action_dim = 4 if self.learn_angle else 2
        self.action_horizon = data_cfg.action_horizon
        self.norm_type = data_cfg.norm_type


        self.is_diffusion_model = True if self.model.name == "pidiff" else False
        if self.is_diffusion_model:
            # For now we use only pidiff model
            noise_scheduler_config = self.model.module.get_scheduler_config() if hasattr(self.model, "module") else self.model.get_scheduler_config()
            self.noise_scheduler = DiffuserScheduler(noise_scheduler_config)

        robot_properties = get_robot_config(robot)[robot]
        robot_properties.update({'frame_rate':self.frame_rate})
        self.action_stats = get_action_stats(properties=robot_properties, waypoint_spacing=waypoint_spacing)

    def load(self, model_name:str, model_version:str = "best_model"):
        """
        Load a pre-trained model.

        Args:
            model_name (str): The name of the pre-trained model.
        """
        
        model_path = os.path.join(CKPTH_PATH, model_name, f"{model_version}.pth")
        checkpoint = torch.load(model_path,map_location='cpu')

        state_dict = checkpoint["model_state_dict"]
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        if self.is_diffusion_model:
            self.noise_scheduler.eval()

    def to(self, device):
        """
        Set the device to which the model should be transferred.

        Args:
            device (torch.device): Device to use.
        """
        self.model.to(device=device)
        self.device = device

    def forward(self, obs_img: torch.Tensor,
                curr_rel_pos_to_target: torch.Tensor = None,
                goal_rel_pos_to_target: torch.Tensor = None,
                prev_actions: torch.Tensor = None,
                vision_mem = None,
                lin_mem = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model to predict the next waypoint.

        Args:
            obs_img (torch.Tensor): Observation image.
            curr_rel_pos_to_target (torch.Tensor): Current relative position to the target.
            goal_rel_pos_to_target (torch.Tensor): Goal relative position to the target.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predicted waypoint and corresponding action.
        """
        
        predicted_waypoints_normalized = self.predict_raw(obs_img,
                                                    curr_rel_pos_to_target,
                                                    goal_rel_pos_to_target,
                                                    prev_actions,
                                                    vision_mem,
                                                    lin_mem)
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
        deltas = unnormalize_data(ndeltas, self.action_stats["pos"],norm_type=self.norm_type)
        waypoints = np.cumsum(deltas, axis=0)
        waypoints = np.concatenate([waypoints, cos_sin_angles],axis=1)

        return waypoints


    def predict_raw(self, obs_img: torch.Tensor,
                    curr_rel_pos_to_target: torch.Tensor,
                    goal_rel_pos_to_target: torch.Tensor,
                    prev_actions: torch.Tensor,
                    vision_mem: torch.Tensor,
                    lin_mem: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        vision_mem_img = vision_mem.unsqueeze(0).to(self.device)
        
        normalized_action_context_queue, target_context_queue, goal_to_target, goal_mask = None, None, None, None
        
        if prev_actions is not None:
            normalized_action_context_queue = actions_forward_pass(prev_actions,self.action_stats,self.learn_angle,norm_type=self.norm_type)
            normalized_action_context_queue = normalized_action_context_queue.unsqueeze(0).to(self.device)
        
        
        if curr_rel_pos_to_target is not None:
            target_context_queue = curr_rel_pos_to_target.unsqueeze(0).to(self.device)
            lin_mem_vec = lin_mem.unsqueeze(0).to(self.device)
            
            # target_context_mask = (torch.sum(curr_rel_pos_to_target==torch.zeros_like(curr_rel_pos_to_target),axis=1) == curr_rel_pos_to_target.shape[1])
            
            # goal_mask = (torch.sum(target_context_mask) == curr_rel_pos_to_target.shape[0]).long()
            
            # not(torch.any(curr_rel_pos_to_target)) -> if there is no target detection in the context -> mask the goal!  
            target_in_context = torch.any(curr_rel_pos_to_target)
            
            # if target in context -> dont mask! , if not -> mask the goal!

            goal_mask = (~target_in_context).long()
            goal_mask = goal_mask.unsqueeze(0).to(self.device)

        if goal_rel_pos_to_target is not None:
            # print(goal_rel_pos_to_target)
            goal_to_target = goal_rel_pos_to_target.unsqueeze(0).to(self.device)

        normalized_actions = self.infer_actions(
                obs_img=context_queue,
                curr_rel_pos_to_target=target_context_queue,
                goal_rel_pos_to_target=goal_to_target,
                goal_mask=goal_mask,
                normalized_action_context = normalized_action_context_queue,
                vision_mem = vision_mem_img,
                lin_mem = lin_mem_vec
            )

        return normalized_actions[0]  # no batch dimension

    @torch.inference_mode() ############################# TODO: refactore here for inference 
    def infer_actions(self, obs_img,
                    curr_rel_pos_to_target,
                    goal_rel_pos_to_target,
                    goal_mask ,
                    normalized_action_context,
                    vision_mem,
                    lin_mem):
        
        if self.is_diffusion_model:
            actions = self.model.infer_actions(
                obs_img,
                curr_rel_pos_to_target,
                goal_rel_pos_to_target, 
                goal_mask ,
                normalized_action_context,
                self.noise_scheduler,
                vision_mem,
                lin_mem
            )
        else:
            actions = self.model.infer_actions(
                obs_img,
                curr_rel_pos_to_target,
                goal_rel_pos_to_target, 
                goal_mask ,
                normalized_action_context
            ) 

        return actions

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
    
    error_arr = [clip_angles(predicted[i]) - clip_angles(target[i]) for i in range(len(predicted))]
    
    return np.mean(error_arr)

def main():
    """
    Performing an example of inference using PilotAgent.
    Loads the model, sets up the dataset, and evaluates the predictions.
    """
    # Set the name of the model to load and evaluate
    log_path = "/home/roblab20/dev/pilot/pilot_bc/pilot_train/logs/train_pilot_policy"
    # model_name = "cnn_mlp_bsz128_c5_ac5_gcFalse_gcp0.1_ph8_tceTrue_ntmaxmin_2024-11-08_12-52-45"
    # model_name = "pidiff_bsz128_c1_ac1_gcTrue_gcp0.5_ph16_tceTrue_ntmaxmin_dnsddpm_2024-11-08_11-13-00"
    # model_name = "pidiff_bsz128_c4_ac4_gcTrue_gcp0.5_ph16_tceTrue_ntmaxmin_dnsddpm_2024-11-08_15-01-23"
    # model_name = "vint_bsz128_c5_ac5_gcTrue_gcp0.1_ph8_tceTrue_ntmaxmin_2024-11-08_13-54-07"
    model_name = "pidiff_bsz16_c1_ac1_gcTrue_gcp0.1_ah16_ph32_tceTrue_ntmaxmin_2024-11-11_12-12-54"
    # model_name = "pidiff_bsz256_c4_ac2_gcTrue_gcp0.3_ph16_tceTrue_ntmaxmin_dnsddpm_2024-11-10_22-59-16"
    
    model_version = "best_model" 
    # Retrieve the model's inference configuration
    data_cfg, datasets_cfg, policy_model_cfg, vision_encoder_cfg, linear_encoder_cfg, device = get_inference_config(model_name=model_name)
    # Define the robot name and retrieve the corresponding dataset configuration
    robot = "go2"
    robot_dataset_cfg = datasets_cfg[robot]
    policy_model = policy_model_cfg.name
    
    # Specify the type of data split to use for testing
    data_split_type = "test"
    eval_type=robot+"_test"
    # Choose the device for computation 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "cuda" else "cpu"

    # Initialize the PilotPlanner model with the appropriate configurations
    wpt_i = 0
    frame_rate = 12
    model = PilotAgent(data_cfg=data_cfg,
                                policy_model_cfg=policy_model_cfg,
                                vision_encoder_cfg=vision_encoder_cfg,
                                linear_encoder_cfg=linear_encoder_cfg,
                                robot=robot,
                                wpt_i=wpt_i,
                                frame_rate=frame_rate)

    # Load the pre-trained model and move to the specified device 
    model.load(model_name=model_name,model_version=model_version)
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

    visualizer = Visualizer(datasets_cfg=datasets_cfg,log_path=log_path)
    
    # Initialize variables for calculating average inference time
    dt_sum = 0
    size = 5000
    
    size = min(size,len(dataset))
    action_horizon = data_cfg.action_horizon
    pos_tot_err = 0
    yaw_tot_err = 0
    
    
    viz_indices = np.random.randint(0,size,size=(1000))
    # Loop through the dataset, performing inference and timing each prediction
    for i in range(0, size):
        # Retrieve relevant data for inference, including context, ground truth actions, etc.
        (
            context_queue,
            target_context_queue,
            goal_to_target,
            gt_actions_normalized,
            unormalized_prev_actions,
            normalized_prev_action,
            normalized_goal_pos,
            unormalized_goal_pos, 
            dataset_index,
            action_mask,
            vision_memory,
            lin_memory) = dataset[i]
        
        viz_images = torch.split(context_queue, 1, dim=0)
        viz_obs_image = TF.resize(viz_images[-1], VIZ_IMAGE_SIZE)
        viz_context_t0_image = TF.resize(viz_images[0], VIZ_IMAGE_SIZE)
        viz_difference = viz_obs_image - viz_context_t0_image
        viz_mem_image = TF.resize(vision_memory, VIZ_IMAGE_SIZE)
        
        # Convert the ground truth actions into waypoints
        gt_waypoints = to_numpy(gt_actions_normalized)
        gt_waypoints = model.get_waypoint(gt_waypoints)
        gt_waypoints = gt_waypoints[:action_horizon,:]
        
        # Start timing the inference process
        t = tic()
        
        # Perform inference to predict the next waypoint
        predicted_waypoint = model(context_queue,
                                target_context_queue,
                                goal_to_target,
                                unormalized_prev_actions,
                                vision_memory,
                                lin_memory)

        # Measure the elapsed time for inference
        dt = toc(t)
        
        
        batch_viz_obs_image = viz_obs_image.unsqueeze(0)
        batch_viz_context_t0_image = viz_context_t0_image.unsqueeze(0)
        batch_goal_pos = unormalized_goal_pos.unsqueeze(0)
        batch_viz_difference = viz_difference.unsqueeze(0)
        
        batch_viz_mem = viz_mem_image.unsqueeze(0)
        batch_dataset_index = dataset_index.unsqueeze(0)
        batch_action_mask = action_mask.unsqueeze(0)
        unormalized_prev_actions = model.get_waypoint(normalized_waypoints=normalized_prev_action)
        batch_unormalized_prev_actions = np.expand_dims(unormalized_prev_actions,axis=0)
        batch_predicted_waypoint = np.expand_dims(predicted_waypoint,axis=0)
        batch_gt_waypoints = np.expand_dims(gt_waypoints,axis=0)

        print(i)
        if i in viz_indices:
            print("visualize...")
            visualizer.visualize_traj_pred_offline(
                    batch_obs_images=to_numpy(batch_viz_obs_image),
                    batch_goal_images = to_numpy(batch_viz_context_t0_image),
                    batch_viz_difference = to_numpy(batch_viz_difference),
                    batch_viz_mem = to_numpy(batch_viz_mem),
                    dataset_indices=to_numpy(batch_dataset_index),
                    batch_goals = to_numpy(batch_goal_pos),
                    batch_pred_waypoints = batch_predicted_waypoint,
                    batch_label_waypoints = batch_gt_waypoints,
                    batch_waypoints_context = batch_unormalized_prev_actions,
                    batch_action_mask=to_numpy(batch_action_mask),
                    policy_model = policy_model,
                    model_name = model_name,
                    model_version = model_version,
                    eval_type=eval_type,
                    normalized=True,
                    n_img=i,
                )



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
    
    results_dir = os.path.join(
                log_path,policy_model,model_name, "visualize", eval_type, "inference_test",model_version
            )
    
    if not os.path.exists(results_dir):
            os.makedirs(results_dir)
    
    results_path = os.path.join(
        results_dir,
        "0_results.json"
    )

    results = {
        "size": size,
        "pos_avg_mse": pos_avg_err,
        "yaw_avg_mse": yaw_avg_err,
        "avg_inference_dt": dt_avg
    }
    
    print(f"inference time avg: {dt_avg}[sec]")
    print(f"Position average MSE Error: {pos_avg_err}")
    print(f"Yaw average Error: {yaw_avg_err} [rad] | {np.rad2deg(yaw_avg_err)} [deg]")
    
    # Save the results to a JSON file
    with open(results_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    print(f"Results saved to {results_path}")


# Entry point for the script to execute the main function
if __name__ == "__main__":
    main()
    