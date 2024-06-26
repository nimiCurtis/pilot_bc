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
    calculate_sin_cos,
    get_data_path,
    to_local_coords,
)

from pilot_utils.utils import (
    get_delta,
    normalize_data
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
        Main Pilot dataset class

        Args:
            data_folder (string): Directory with all the image data
            data_split_folder (string): Directory with filepaths.txt, a list of all trajectory names in the dataset split that are each seperated by a newline
            dataset_name (string): Name of the dataset [recon, go_stanford, scand, tartandrive, etc.]
            waypoint_spacing (int): Spacing between waypoints
            min_dist_cat (int): Minimum distance category to use
            max_dist_cat (int): Maximum distance category to use
            negative_mining (bool): Whether to use negative mining from the ViNG paper (Shah et al.) (https://arxiv.org/abs/2012.09812)
            len_traj_pred (int): Length of trajectory of waypoints to predict if this is an action dataset
            learn_angle (bool): Whether to learn the yaw of the robot at each predicted waypoint if this is an action dataset
            context_size (int): Number of previous observations to use as context
            context_type (str): Whether to use temporal, randomized, or randomized temporal context
            end_slack (int): Number of timesteps to ignore at the end of the trajectory
            goals_per_obs (int): Number of goals to sample per observation
            normalize (bool): Whether to normalize the distances or actions
            goal_type (str): What data type to use for the goal. The only one supported is "image" for now.
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
        self.len_traj_pred=data_cfg.len_traj_pred
        self.target_context=data_cfg.target_context
        self.learn_angle=data_cfg.learn_angle
        
        if self.learn_angle:
            self.num_action_params = 3
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
        state = self.__dict__.copy() # Make a copy of the object's dictionary.
        state["_image_cache"] = None # Explicitly remove the image cache from the serialized state.
        return state 
    
    def __setstate__(self, state):
        self.__dict__ = state # Restore the serialized state.
        self._build_caches() # Rebuild the image cache after the object has been deserialized.

    def _build_caches(self, use_tqdm: bool = True):
        """
        Build a cache of images for faster loading using LMDB
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
        Build an index consisting of tuples (trajectory name, time, max goal distance)
        """
        samples_index = []
        goals_index = []

        for traj_name in tqdm.tqdm(self.traj_names, disable=not use_tqdm, dynamic_ncols=True):
            traj_data = self._get_trajectory(traj_name)
            properties = self._get_properties(traj_name) ## added
            traj_len = len(traj_data["position"])
            
            # Check robot and target data length is equal
            if self.goal_condition:
                target_traj_data = self._get_trajectory(traj_name, target=True)
                target_traj_len = len(target_traj_data)
                assert traj_len == target_traj_len, "robot and traget data length not equal"


            for goal_time in range(0, traj_len):
                goals_index.append((traj_name, goal_time))

            begin_time = self.context_size * self.waypoint_spacing
            end_time = traj_len - self.end_slack - self.len_traj_pred * self.waypoint_spacing
            for curr_time in range(begin_time, end_time):
                max_goal_distance = min(self.max_dist_cat*self.waypoint_spacing, traj_len - curr_time - 1) # keep max distance in range 

                samples_index.append((traj_name, properties, curr_time, max_goal_distance))

        return samples_index, goals_index

    def _sample_goal(self, trajectory_name, curr_time, max_goal_dist):
        """
        Sample a goal from the future in the same trajectory.
        Returns: (trajectory_name, goal_time, goal_is_negative)
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
        Sample a goal from a (likely) different trajectory.
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
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred * self.waypoint_spacing + 1
        yaw = np.array(traj_data["yaw"][start_index:end_index:self.waypoint_spacing])
        positions = np.array(traj_data["position"][start_index:end_index:self.waypoint_spacing])
        goal_pos = np.array(traj_data["position"][min(goal_time, len(traj_data["position"]) - 1)])

        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)

        if yaw.shape != (self.len_traj_pred + 1,):
            const_len = self.len_traj_pred + 1 - yaw.shape[0]
            yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
            positions = np.concatenate([positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0)

        assert yaw.shape == (self.len_traj_pred + 1,), f"{yaw.shape} and {(self.len_traj_pred + 1,)} should be equal"
        assert positions.shape == (self.len_traj_pred + 1, 2), f"{positions.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

        waypoints = to_local_coords(positions, positions[0], yaw[0])
        goal_in_local = to_local_coords(goal_pos, positions[0],yaw[0])
        
        assert waypoints.shape == (self.len_traj_pred + 1, 2), f"{waypoints.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

        if self.learn_angle:
            # shape reduce from self.len_traj_pred + 1 to self.len_traj_pred
            yaw = yaw[1:] - yaw[0] # yaw is relative to the current yaw # already a cumsum
            actions = np.concatenate([waypoints[1:], yaw[:, None]], axis=-1) 
        else:
            actions = waypoints[1:]

        if self.normalize: 
            #Ours
            ## only for pos
            actions_deltas = get_delta(actions[:, :2])
            normalized_actions_deltas = normalize_data(actions_deltas,action_stats['pos'])
            actions[:, :2] = np.cumsum(normalized_actions_deltas, axis=0)

            # goal in local is already the delta from the current pos
            normalized_goal_delta = normalize_data(goal_in_local, action_stats['pos'])
            goal_pos = normalized_goal_delta

            #VinT
            #actions[:, :2] /= self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing # TODO: depend on data
            #goal_pos = self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing

        assert actions.shape == (self.len_traj_pred, self.num_action_params), f"{actions.shape} and {(self.len_traj_pred, self.num_action_params)} should be equal"

        return actions, goal_pos
    
    def _get_trajectory(self, trajectory_name, target:bool = False):
        
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
        return len(self.index_to_data)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        """
        Args:
            i (int): index to ith datapoint
        Returns:
            Tuple of tensors containing the context, observation, goal, transformed context, transformed observation, transformed goal, distance label, and action label
                obs_image (torch.Tensor): tensor of shape [3, H, W] containing the image of the robot's observation
                goal_image (torch.Tensor): tensor of shape [3, H, W] containing the subgoal image 
                action_label (torch.Tensor): tensor of shape (5, 2) or (5, 4) (if training with angle) containing the action labels from the observation to the goal
                which_dataset (torch.Tensor): index of the datapoint in the dataset [for identifying the dataset for visualization when using multiple datasets]
        """

        # self.index_to_data[i] = (traj_name, curr_time, max_goal_distance)
        f_curr, curr_properties, curr_time, max_goal_dist = self.index_to_data[i]
        f_goal, goal_time, goal_is_negative = self._sample_goal(f_curr, curr_time, max_goal_dist)
        # goal is negative -> probably a goal pos from another trajectory. 

        # Load images
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
            
            # context_times = list(
            #     range(
            #         curr_time + -self.context_size,
            #         curr_time + 1,
            #         1,
            #     )
            # )
            
            context = [(f_curr, t) for t in context_times]
        else:
            raise ValueError(f"Invalid context type {self.context_type}")

        # Load context images
        obs_image = [
                self._load_image(f, t) for f, t in context
            ]
        
        # Transform
        obs_image = transform_images(obs_image, self.transform)

        # Load other trajectory data
        curr_traj_data = self._get_trajectory(f_curr)
        curr_traj_len = len(curr_traj_data["position"])
        assert curr_time < curr_traj_len, f"{curr_time} >= {curr_traj_len}"

        # Compute actions
        action_stats = self._get_action_stats(curr_properties,self.waypoint_spacing) ## added
        
        actions, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time, action_stats)
        #actions = waypoints, goal_pos = the position at the goal woth relate to current position
        
        
        if self.goal_condition:
            # Load current position rel to target. use f_curr and curr_time.

            curr_target_traj_data = self._get_trajectory(f_curr, target=True)

            # Load goal position relative to target

            goal_target_traj_data = self._get_trajectory(f_goal, target=True)
            goal_target_traj_data_len = len(goal_target_traj_data)
            # goal_time = min(goal_time, goal_target_traj_data_len-1)
            assert goal_time < goal_target_traj_data_len, f"{goal_time} an {goal_target_traj_data_len}"
            goal_rel_pos_to_target = np.array(goal_target_traj_data[goal_time]["position"][:2]) # Takes the [x,y] 
        
        
            # Take context of target rel pos or only the recent
            if self.target_context:
                np_curr_rel_pos_to_target = np.array([
                    curr_target_traj_data[t]["position"][:2] for f, t in context
                ])
                
                # Normalizing each column independently
                max_val_col = np_curr_rel_pos_to_target.max(axis=0)
                min_val_col = np_curr_rel_pos_to_target.min(axis=0)
                
                normalized_array_separate = 2 * (np_curr_rel_pos_to_target - min_val_col) / (max_val_col - min_val_col) - 1
                
                normalized_array_separate[:-1] =  normalized_array_separate[:-1] - normalized_array_separate[-1]

            else:
                np_curr_rel_pos_to_target = np.array(curr_target_traj_data[curr_time]["position"][:2]) # Takes the [x,y] 

            
            # Take the deltas to the goal relative position to the target
            goal_rel_pos_to_target = 2 * (goal_rel_pos_to_target - min_val_col) / (max_val_col - min_val_col) - 1
            goal_rel_pos_to_target = goal_rel_pos_to_target - normalized_array_separate[-1]
            
            # Cat and tensor the context of relative positions to target
            curr_rel_pos_to_target = torch.flatten(torch.as_tensor(normalized_array_separate))

        else:
            curr_rel_pos_to_target = np.zeros_like((actions.shape[0],2))
            goal_rel_pos_to_target = np.array([0,0])

        # Compute timesteps distances
        if goal_is_negative:
            distance = self.max_dist_cat
        else:
            distance = (goal_time - curr_time) // self.waypoint_spacing
            assert (goal_time - curr_time) % self.waypoint_spacing == 0, f"{goal_time} and {curr_time} should be separated by an integer multiple of {self.waypoint_spacing}, target_traj_len = {goal_target_traj_data_len}"

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

    ## added
    def _get_properties(self,trajectory_name):

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

    ## added
    def _get_action_stats(self,properties, waypoint_spacing):
        
        frame_rate = properties['frame_rate']
        lin_vel_lim = properties['max_lin_vel']
        ang_vel_lim = properties['max_ang_vel']
        
        return {'pos': {'max': (lin_vel_lim / frame_rate)*waypoint_spacing,
                        'min': -(lin_vel_lim /frame_rate)*waypoint_spacing},
                'yaw': {'max': (ang_vel_lim /frame_rate)*waypoint_spacing,
                        'min': -(ang_vel_lim /frame_rate)*waypoint_spacing }}
