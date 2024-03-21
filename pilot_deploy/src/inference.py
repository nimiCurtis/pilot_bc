
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import yaml
from PIL import Image as PILImage
from typing import List
from pilot_train.models.vint.vint import ViNT
from pilot_train.data.pilot_dataset import PilotDataset
from torchvision import transforms

from .utils import (tic, toc, unnormalize_data,
                    to_numpy,
                    transform_images,
                    get_delta,
                    calculate_sin_cos)

class InferenceDataset(PilotDataset):
    def __init__(self, config_path):
        
        with open(config_path,"r") as config_file:
            self.config = yaml.safe_load(config_file)

        dataset_name = "turtlebot"
        data_config = self.config["datasets"][dataset_name]
        data_type = "test"
        super().__init__(data_folder=data_config["data_folder"],
                        data_split_folder=data_config[data_type],
                        dataset_name=dataset_name,
                        image_size=self.config["image_size"],
                        waypoint_spacing=1,
                        min_dist_cat=self.config["distance"]["min_dist_cat"],
                        max_dist_cat=self.config["distance"]["max_dist_cat"],
                        min_action_distance=self.config["action"]["min_dist_cat"],
                        max_action_distance=self.config["action"]["max_dist_cat"],
                        negative_mining=data_config["negative_mining"],
                        len_traj_pred=self.config["len_traj_pred"],
                        learn_angle=self.config["learn_angle"],
                        context_size=self.config["context_size"],
                        context_type=self.config["context_type"],
                        end_slack=data_config["end_slack"],
                        goals_per_obs=data_config["goals_per_obs"],
                        normalize=self.config["normalize"],
                        goal_type=self.config["goal_type"],)
    
    
    def __getitem__(self, i: int) -> List[PILImage.Image]:
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
        print(f"collect from: {f_curr} | sample number: {curr_time}")
        f_goal, goal_time, goal_is_negative = self._sample_goal(f_curr, curr_time, max_goal_dist)
        # goal is negative ??? TODO : check

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
            context = [(f_curr, t) for t in context_times]
        else:
            raise ValueError(f"Invalid context type {self.context_type}")

        obs_image = torch.cat([
            self._load_image(f, t) for f, t in context
        ])

        # Load goal image
        goal_image = self._load_image(f_goal, goal_time)

        # Load other trajectory data
        curr_traj_data = self._get_trajectory(f_curr)
        curr_traj_len = len(curr_traj_data["position"])
        assert curr_time < curr_traj_len, f"{curr_time} and {curr_traj_len}"

        goal_traj_data = self._get_trajectory(f_goal)
        goal_traj_len = len(goal_traj_data["position"])
        assert goal_time < goal_traj_len, f"{goal_time} an {goal_traj_len}"

        # Compute actions
        action_stats = self._get_action_stats(curr_properties,self.waypoint_spacing) ## added
        ###### here setting the goal time to constant
        #actions, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time)
        actions, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time, action_stats)

        # Compute distances
        if goal_is_negative:
            distance = self.max_dist_cat
        else:
            distance = (goal_time - curr_time) // self.waypoint_spacing
            assert (goal_time - curr_time) % self.waypoint_spacing == 0, f"{goal_time} and {curr_time} should be separated by an integer multiple of {self.waypoint_spacing}"
        
        actions_torch = torch.as_tensor(actions, dtype=torch.float32)
        if self.learn_angle:
            actions_torch = calculate_sin_cos(actions_torch)

        return (
            torch.as_tensor(obs_image, dtype=torch.float32), # [C*(context+1),H,W]
            actions_torch,  # [trej_len_pred,4]
            torch.as_tensor(self.dataset_index, dtype=torch.int64),
        )



class InferenceModel:
    def __init__(self, config_path):
        """
        Initializes the InferenceModel with the path to the model weights and configuration.

        :param model_path: Path to the model weights file.
        :param config_path: Path to the configuration file.
        """
        
        self.transform = ([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transform = transforms.Compose(self.transform)
        
        
        with open(config_path,"r") as config_file:
            config = yaml.safe_load(config_file)
        
        self.wpt_id = config["wpt_id"]
        model_config_path = config["config_path"]
            
        with open(model_config_path,"rb") as model_config_file:
            self.model_config = yaml.safe_load(model_config_file)

        robot_properties = config["properties"]
        self.action_stats = self._get_action_stats(robot_properties)

        # Load the model
        model_path = config["model_path"]
        device = config["device"] 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "cuda" else "cpu"
        self.model = self.load_model(model_path)

    def load_model(self, model_path) -> nn.Module:
        """
        Loads the model from the model_path.

        This method should be implemented to load the model using the framework of your choice (e.g., PyTorch, TensorFlow).

        :return: The loaded model.
        """

        model_type = self.model_config["model_type"]

        if model_type == "vint":
            model = ViNT(
                context_size=self.model_config["context_size"],
                len_traj_pred=self.model_config["len_traj_pred"],
                learn_angle=self.model_config["learn_angle"],
                obs_encoder=self.model_config["obs_encoder"],
                obs_encoding_size=self.model_config["obs_encoding_size"],
                late_fusion=self.model_config["late_fusion"],
                mha_num_attention_heads=self.model_config["mha_num_attention_heads"],
                mha_num_attention_layers=self.model_config["mha_num_attention_layers"],
                mha_ff_dim_factor=self.model_config["mha_ff_dim_factor"],
                goal_condition=False #model_config["goal_condition"]
            )
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        checkpoint = torch.load(model_path, map_location=self.device)
        loaded_model = checkpoint["model"]
        try:
            state_dict = loaded_model.module.state_dict()
            model.load_state_dict(state_dict, strict=False)
        except AttributeError as e:
            state_dict = loaded_model.state_dict()
            model.load_state_dict(state_dict, strict=False)

        model.to(self.device)
        model.eval()
        return model

    def predict(self, context_queue):
        """
        Runs inference on the input data and returns the prediction.

        :param input_data: The input data for the model.
        :return: The prediction made by the model.
        """

        context_queue_torch = transform_images(context_queue, self.model_config["image_size"])
        # context_queue_torch = context_queue.unsqueeze(0)
        # has no meaning when goal condition=false
        context_queue_torch = context_queue_torch.to(self.device)
        obs_images = torch.split(context_queue_torch, 3, dim=1)
        obs_images = torch.cat(obs_images, dim=1)

        ## has no meaning
        goal_image = obs_images[-1][:3].unsqueeze(0)

        # infer model
        with torch.no_grad():
            distances, torch_normalized_waypoints = self.model(obs_images, goal_image)

        # to numpy
        distances = to_numpy(distances)
        normalized_waypoints = to_numpy(torch_normalized_waypoints)

        waypoint = self._get_waypoint(normalized_waypoints)
        
        return waypoint

    def _get_action_stats(self,properties):
            frame_rate = properties['frame_rate']
            lin_vel_lim = properties['max_lin_vel']
            ang_vel_lim = properties['max_ang_vel']

            return {'pos': {'max': (lin_vel_lim / frame_rate),
                            'min': -(lin_vel_lim /frame_rate)},
                    'yaw': {'max': (ang_vel_lim /frame_rate),
                            'min': -(ang_vel_lim /frame_rate)}}

    def _get_waypoint(self,normalized_waypoints):

        # get delta from current position
        cos_sin_angle = normalized_waypoints[0][:,2:][self.wpt_id]
        ndeltas = get_delta(normalized_waypoints[0][:,:2])
        # ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2) # ??
        ndeltas = unnormalize_data(ndeltas, self.action_stats["pos"])
        waypoints = np.cumsum(ndeltas, axis=0)
        waypoint = waypoints[self.wpt_id]
        waypoint = np.concatenate([waypoint,cos_sin_angle])

        return waypoint


def main():

    data_config_path = "/home/roblab20/dev/pilot/pilot_bc/pilot_train/config/pilot.yaml"
    dataset = InferenceDataset(data_config_path)
    
    config_path = "/home/roblab20/dev/pilot/pilot_bc/pilot_deploy/config/config.yaml"
    model = InferenceModel(config_path=config_path)
    
    dt_sum = 0
    size = 800
    # size = len(dataset)
    for i in range(200,size):
        context_queue, gt_waypoints, dataset_index = dataset[i]
        
        gt_waypoints = to_numpy(gt_waypoints)
        gt_waypoints = np.array([gt_waypoints])
        gt_waypoint = model._get_waypoint(gt_waypoints)

        t = tic()
        waypoint = model.predict(context_queue)
        dt = toc(t)
        print(f"infer: dataset index: {dataset_index} | sample: {i}")
        print(f"wpt predicted: {np.round(waypoint,5)}")
        print(f"wpt gt: {np.round(gt_waypoint,5)}")
        print(f"inference time: {dt}[sec]")
        print()
        if(i>0):
            dt_sum+=dt
    dt_avg = dt_sum / (size-1)
    print(f"inference time avg: {dt_avg}[sec]")

if __name__ == "__main__":
    main()
    