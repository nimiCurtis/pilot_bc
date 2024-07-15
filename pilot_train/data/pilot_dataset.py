import numpy as np
import os
import json
from typing import Any, Dict, List, Tuple
import tqdm
import io
import lmdb
from omegaconf import DictConfig

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from pilot_utils.data.data_utils import (
    img_path_to_data,
    get_data_path,
    to_local_coords,
)

from pilot_utils.utils import (
    get_delta,
    normalize_data,
    xy_to_d_cos_sin,
    actions_forward_pass,
    get_action_stats
)

from pilot_utils.transforms import transform_images


from pilot_config.config import get_robot_config, get_recording_config
class PilotDataset(Dataset):
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
        Initializes the PilotDataset class.

        Args:
            data_cfg (DictConfig): Configuration for the data, including goal conditions, image size, and normalization parameters.
            datasets_cfg (DictConfig): Configuration for the datasets, including data folders and maximum depth.
            robot_dataset_cfg (DictConfig): Configuration specific to the robot dataset, including waypoint spacing and negative mining options.
            dataset_name (str): Name of the dataset (e.g., 'recon', 'go_stanford', 'scand', 'tartandrive').
            data_split_type (str): Type of data split (e.g., 'train', 'test', 'val').
            transform (transforms, optional): Transformations to apply to the data. Default is None.
        """

        # Robot cfg
        data_split_folder=robot_dataset_cfg[data_split_type]
        self.waypoint_spacing=robot_dataset_cfg.waypoint_spacing
        self.negative_mining=robot_dataset_cfg.negative_mining
        self.end_slack=robot_dataset_cfg.end_slack
        self.goals_per_obs=robot_dataset_cfg.goals_per_obs
        self.data_config = robot_dataset_cfg

        # Data cfg
        self.goal_condition = data_cfg.goal_condition

        self.image_size= data_cfg.image_size
        self.normalize=data_cfg.normalize
        self.goal_type=data_cfg.goal_type
        self.obs_type = data_cfg.obs_type
        self.img_type = data_cfg.img_type 
        self.pred_horizon=data_cfg.pred_horizon
        self.target_context=data_cfg.target_context
        self.learn_angle=data_cfg.learn_angle
        
        if self.learn_angle:
            self.num_action_params = 4
        else:
            self.num_action_params = 2
        self.context_size=data_cfg.context_size
        assert data_cfg.context_type in {
            "temporal",
            "randomized",
            "randomized_temporal",
        }, "context_type must be one of temporal, randomized, randomized_temporal"
        self.context_type = data_cfg.context_type
        
        # Possible distances for predicting distance
        self.distance_categories = list(
            range(data_cfg.distance.min_dist_cat,
                data_cfg.distance.max_dist_cat + 1,
                self.waypoint_spacing)
        )
        self.min_dist_cat = self.distance_categories[0]
        self.max_dist_cat = self.distance_categories[-1]
        ## I dont think it has a meaning
        if self.negative_mining:
            self.distance_categories.append(-1)
        self.min_action_distance=data_cfg.action.min_dist_cat
        self.max_action_distance=data_cfg.action.max_dist_cat

        # Names and folders
        data_folder=datasets_cfg.data_folder
        self.data_folder = data_folder
        self.data_split_folder = data_split_folder
        self.dataset_name = dataset_name
        
        self.max_depth = datasets_cfg.max_depth
        
        # Load trajectories names for this robot
        traj_names_file = os.path.join(data_split_folder, "traj_names.txt")
        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            self.traj_names = file_lines.split("\n")
        if "" in self.traj_names:
            self.traj_names.remove("")

        # Organize data and indexing
        dataset_names = datasets_cfg.robots
        dataset_names.sort()
        
        
        # Tranform
        self.transform = transform
        
        # Use this index to retrieve the dataset name from the dataset config
        self.dataset_index = dataset_names.index(self.dataset_name)
        self.trajectory_cache = {}
        self.target_trajectory_cache = {}
        self._load_index()
        self._build_caches()

    def __getstate__(self):
        """
        Prepares the object's state for serialization.

        Returns:
            dict: The state of the object.
        """
        state = self.__dict__.copy() # Make a copy of the object's dictionary.
        state["_image_cache"] = None # Explicitly remove the image cache from the serialized state.
        return state 
    
    def __setstate__(self, state):
        """
        Restores the object's state from serialization.

        Args:
            state (dict): The state to restore.
        """
        self.__dict__ = state # Restore the serialized state.
        self._build_caches() # Rebuild the image cache after the object has been deserialized.

    def _build_caches(self, use_tqdm: bool = True):
        """
        Builds a cache of images for faster loading using LMDB.

        Args:
            use_tqdm (bool): Whether to use tqdm for progress visualization. Default is True.
        """
        
        cache_filename = os.path.join(
            self.data_split_folder,
            f"dataset_{self.dataset_name}_context_{self.img_type}.lmdb",
        )

        # Load all the trajectories into memory. These should already be loaded, but just in case.
        for traj_name in self.traj_names:
            self._get_trajectory(traj_name)
            if self.goal_condition:
                self._get_trajectory(traj_name, target=True)

        # TODO: load target trajectories into memory.

        # If the cache file doesn't exist, create it by iterating through the dataset and writing each image to the cache
        if not os.path.exists(cache_filename):
            tqdm_iterator = tqdm.tqdm(
                self.goals_index,
                disable=not use_tqdm,
                dynamic_ncols=True,
                desc=f"Building LMDB cache for {self.dataset_name}"
            )

            # map_size = 2**40 bytes = 1 TB
            with lmdb.open(cache_filename, map_size=2**40) as image_cache:
                with image_cache.begin(write=True) as txn:
                    for traj_name, time in tqdm_iterator:
                        image_path = get_data_path(self.data_folder,self.img_type, traj_name, time)
                        with open(image_path, "rb") as f:
                            txn.put(image_path.encode(), f.read())

        # Reopen the cache file in read-only mode
        self._image_cache: lmdb.Environment = lmdb.open(cache_filename, readonly=True)

    def _build_index(self, use_tqdm: bool = False):
        """
        Builds an index consisting of tuples (trajectory name, time, max goal distance).

        Args:
            use_tqdm (bool): Whether to use tqdm for progress visualization. Default is False.

        Returns:
            tuple: Two lists, one for samples index and one for goals index.
        """
        
        samples_index = []
        goals_index = []

        # Iterate through each trajectory name with optional progress visualization using tqdm
        for traj_name in tqdm.tqdm(self.traj_names, disable=not use_tqdm, dynamic_ncols=True):
            # Retrieve trajectory data and properties for the current trajectory
            traj_data = self._get_trajectory(traj_name)
            properties = self._get_properties(traj_name)
            traj_len = len(traj_data["position"])
            
            # Check that the lengths of robot and target data are equal if goal_condition is true
            if self.goal_condition:
                target_traj_data = self._get_trajectory(traj_name, target=True)
                target_traj_len = len(target_traj_data)
                assert traj_len == target_traj_len, "Robot and target data lengths are not equal"

            # Add all possible goal times for the current trajectory to the goals index
            for goal_time in range(0, traj_len):
                goals_index.append((traj_name, goal_time))

            # Define the range for current times based on context size, end slack, and prediction horizon
            begin_time = self.context_size * self.waypoint_spacing
            end_time = traj_len - self.end_slack - self.pred_horizon * self.waypoint_spacing

            # Add samples to the samples index with their respective max goal distances
            for curr_time in range(begin_time, end_time):
                max_goal_distance = min(self.max_dist_cat * self.waypoint_spacing, traj_len - curr_time - 1)  # Keep max distance in range
                samples_index.append((traj_name, properties, curr_time, max_goal_distance))

        # Return the constructed samples index and goals index
        return samples_index, goals_index

    def _sample_goal(self, trajectory_name, curr_time, max_goal_dist):
        """
        Samples a goal from the future in the same trajectory.

        Args:
            trajectory_name (str): Name of the trajectory.
            curr_time (int): Current time step.
            max_goal_dist (int): Maximum goal distance.

        Returns:
            tuple: The trajectory name, goal time, and a boolean indicating if the goal is negative.
        """
        
        goal_offset = np.random.randint(0, (max_goal_dist/self.waypoint_spacing) + 1)
        if goal_offset == 0:
            trajectory_name, goal_time = self._sample_negative()
            return trajectory_name, goal_time, True
        else:
            goal_time = curr_time + goal_offset*self.waypoint_spacing
            return trajectory_name, goal_time, False

    def _sample_negative(self):
        """
        Samples a goal from a (likely) different trajectory.

        Returns:
            tuple: The trajectory name and goal time.
        """
        
        return self.goals_index[np.random.randint(0, len(self.goals_index))]

    def _load_index(self) -> None:
        """
        Generates a list of tuples of (obs_traj_name, goal_traj_name, obs_time, goal_time) for each observation in the dataset
        """
        index_to_data_path = os.path.join(
            self.data_split_folder,
            f"dataset_dist_{self.min_dist_cat}_to_{self.max_dist_cat}_context_{self.context_type}_obs_{self.img_type}_n{self.context_size}_slack_{self.end_slack}.json",
        )
        try:
            # load the index_to_data if it already exists (to save time)
            with open(index_to_data_path, "rb") as f:
                self.index_to_data, self.goals_index = json.load(f)
        except:
            # if the index_to_data file doesn't exist, create it
            self.index_to_data, self.goals_index = self._build_index()
            with open(index_to_data_path, "w") as f:
                json.dump((self.index_to_data, self.goals_index), f)

    def _load_image(self, trajectory_name, time):
        """
        Loads an image from the cache.

        Args:
            trajectory_name (str): Name of the trajectory.
            time (int): Time step.

        Returns:
            Image: The loaded image.
        """
        
        image_path = get_data_path(self.data_folder, self.img_type, trajectory_name, time)

        try:
            with self._image_cache.begin() as txn:
                image_buffer = txn.get(image_path.encode())
                image_bytes = bytes(image_buffer)
            image_bytes = io.BytesIO(image_bytes)
            return img_path_to_data(image_bytes, self.image_size)
        except TypeError:
            print(f"Failed to load image {image_path}")

        
    def _compute_actions(self, traj_data, curr_time, goal_time, action_stats):
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
        end_index = curr_time + self.pred_horizon * self.waypoint_spacing + 1

        # Extract yaw and position data from the trajectory
        yaw = np.array(traj_data["yaw"][start_index:end_index:self.waypoint_spacing])
        positions = np.array(traj_data["position"][start_index:end_index:self.waypoint_spacing])

        # Get the goal position, ensuring it does not exceed the length of the trajectory
        goal_pos = np.array(traj_data["position"][min(goal_time, len(traj_data["position"]) - 1)])

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

        # Ensure waypoints have the correct shape
        assert waypoints.shape == (self.pred_horizon + 1, 2), f"{waypoints.shape} and {(self.pred_horizon + 1, 2)} should be equal"

        if self.learn_angle:
            # Compute relative yaw changes and concatenate with waypoints
            yaw = yaw[1:] - yaw[0]  # yaw is relative to the initial yaw
            actions = np.concatenate([waypoints[1:], yaw[:, None]], axis=-1)
        else:
            actions = waypoints[1:]

        if self.normalize:
            # Normalize the actions based on provided action statistics
            normalized_actions = actions_forward_pass(actions, action_stats, self.learn_angle)

            # Normalize the goal position in local coordinates
            normalized_goal_pos = normalize_data(goal_in_local, action_stats['pos'])

        # Assertion to ensure the shape of normalized actions is correct
        assert normalized_actions.shape == (self.pred_horizon, self.num_action_params), f"{normalized_actions.shape} and {(self.pred_horizon, self.num_action_params)} should be equal"

        return normalized_actions, normalized_goal_pos
    
    def _get_trajectory(self, trajectory_name, target:bool = False):
        """
        Retrieves trajectory data from the cache or loads it from disk.

        Args:
            trajectory_name (str): Name of the trajectory.
            target (bool): Whether to load target trajectory data. Default is False.

        Returns:
            dict: Trajectory data.
        """
        
        if not target:
            if trajectory_name in self.trajectory_cache:
                return self.trajectory_cache[trajectory_name]
            else:
                with open(os.path.join(self.data_folder, trajectory_name, "traj_robot_data.json"), "rb") as f:
                    traj_data = json.load(f)
                self.trajectory_cache[trajectory_name] = traj_data['odom_frame']
                return traj_data['odom_frame']

        else:

            if trajectory_name in self.target_trajectory_cache:
                return self.target_trajectory_cache[trajectory_name]
            else:
                with open(os.path.join(self.data_folder, trajectory_name, "traj_target_data.json"), "rb") as f:
                    traj_data = json.load(f)
                self.target_trajectory_cache[trajectory_name] = traj_data
                return traj_data

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.index_to_data)

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

    def _get_properties(self,trajectory_name):
        """
        Retrieves the properties of the robot for the given trajectory.

        Args:
            trajectory_name (str): Name of the trajectory.

        Returns:
            dict: Robot properties.
        """
        
        recording_config = get_recording_config(data_folder=self.data_folder,
                                                trajectory_name=trajectory_name)
        robot = recording_config['demonstrator']
        frame_rate = recording_config['sync_rate']

        robot_properties = get_robot_config(robot)
        lin_vel_lim = robot_properties[robot]['max_lin_vel']
        ang_vel_lim = robot_properties[robot]['max_ang_vel']

        return {
            'robot': robot,
            'frame_rate': frame_rate,
            'max_lin_vel': lin_vel_lim,
            'max_ang_vel': ang_vel_lim
        }

