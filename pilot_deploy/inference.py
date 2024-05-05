
import os
from omegaconf import DictConfig
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import yaml
from PIL import Image as PILImage
from typing import List, Tuple
from torchvision import transforms

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

CKPTH_PATH = os.path.join(os.path.dirname(__file__),
                        "checkpoints")


class InferenceDataset(PilotDataset):
    def __init__(self, data_cfg: DictConfig,
                datasets_cfg: DictConfig,
                robot_dataset_cfg: DictConfig,
                dataset_name: str,
                data_split_type: str):
        super().__init__(data_cfg,
                        datasets_cfg,
                        robot_dataset_cfg,
                        dataset_name,
                        data_split_type)
        
        def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
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
            #actions = waypoints, goal_pos = the position at the goal woth relate to current position
            
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

class PilotPlanner(nn.Module):
    def __init__(self,
                data_cfg,
                policy_model_cfg,
                encoder_model_cfg,
                robot):
        
        super().__init__()
        
        self.model = get_policy_model(policy_model_cfg=policy_model_cfg,
                                    encoder_model_cfg=encoder_model_cfg,
                                    data_cfg=data_cfg)

        robot_properties = get_robot_config(robot)[robot]
        self.action_stats = self._get_action_stats(robot_properties)

    def load(self, model_name):
        
        model_path = os.path.join(CKPTH_PATH,
                                model_name,
                                "best_model.pth")
        
        checkpoint = torch.load(model_path)

        
        # try:
        state_dict = checkpoint["model_state_dict"]
        self.model.load_state_dict(state_dict, strict=False)
        # except AttributeError as e:


        self.model.eval()


    def to(self,device):
        self.model.to(device=device)
        self.device = device

    def forward(
                self, obs_img: torch.tensor, curr_rel_pos_to_target: torch.tensor, goal_rel_pos_to_target: torch.tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        

        predicted_waypoints_normalized = self.predict_raw(obs_img, curr_rel_pos_to_target, goal_rel_pos_to_target)
        predicted_waypoints_normalized = to_numpy(predicted_waypoints_normalized)
        predicted_waypoint = self.get_waypoint(predicted_waypoints_normalized, wpt_i=2)
        
        return predicted_waypoint

    def get_waypoint(self,normalized_waypoints, wpt_i:int=None):
        # get delta from current position
        cos_sin_angle = normalized_waypoints[:,2:][wpt_i]
        ndeltas = get_delta(normalized_waypoints[:,:2])
        # ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2) # ??
        deltas = unnormalize_data(ndeltas, self.action_stats["pos"])
        waypoints = np.cumsum(deltas, axis=0)
        waypoint = waypoints[wpt_i]
        waypoint = np.concatenate([waypoint,cos_sin_angle])
        
        return waypoint

    ## added
    def _get_action_stats(self,properties):
        
        frame_rate = 6
        lin_vel_lim = properties['max_lin_vel']
        ang_vel_lim = properties['max_ang_vel']
        
        return {'pos': {'max': (lin_vel_lim / frame_rate),
                        'min': -(lin_vel_lim /frame_rate)},
                'yaw': {'max': (ang_vel_lim /frame_rate),
                        'min': -(ang_vel_lim /frame_rate)}}

    def predict_raw(
                self, obs_img: torch.tensor, curr_rel_pos_to_target: torch.tensor, goal_rel_pos_to_target: torch.tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        context_queue = obs_img.unsqueeze(0).to(self.device)
        target_context_queue = curr_rel_pos_to_target.unsqueeze(0).to(self.device)
        goal_to_target = goal_rel_pos_to_target.unsqueeze(0).to(self.device)

        with torch.no_grad():
            normalized_actions = self.model(context_queue, target_context_queue, goal_to_target)
        
        return normalized_actions[0] # no batch dim

def get_inference_config(model_name):
    model_source_dir = os.path.join(CKPTH_PATH, model_name)
    return get_inference_model_config(model_source_dir, rt=True)



def main():

    model_name = "pilot-turtle-static-follower_2024-05-02_12-38-32"
    data_cfg, datasets_cfg, policy_model_cfg, encoder_model_cfg, device = get_inference_config(model_name=model_name)
    robot = "turtlebot"
    robot_dataset_cfg = datasets_cfg[robot]
    data_split_type = "test"
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "cuda" else "cpu"
    
    model = PilotPlanner(data_cfg=data_cfg,
                        policy_model_cfg=policy_model_cfg,
                        encoder_model_cfg=encoder_model_cfg,
                        robot=robot)
    
    model.load(model_name=model_name)
    model.to(device=device)

    dataset = InferenceDataset(data_cfg=data_cfg,
                            datasets_cfg=datasets_cfg,
                            robot_dataset_cfg=robot_dataset_cfg,
                            dataset_name=robot,
                            data_split_type = data_split_type)

    dt_sum = 0
    size= 2000
    for i in range(0,size):
        context_queue, target_context_queue, goal_to_target, gt_actions_normalized, _, dataset_index, _ = dataset[i]
        gt_waypoints = to_numpy(gt_actions_normalized)
        gt_waypoint = model.get_waypoint(gt_waypoints, wpt_i=2)

        t = tic()
        predicted_waypoint = model(context_queue, target_context_queue, goal_to_target)
        dt = toc(t)
        
        print(f"infer: dataset index: {dataset_index} | sample: {i}")
        print(f"wpt predicted: {np.round(predicted_waypoint,5)}")
        print(f"wpt gt: {np.round(gt_waypoint,5)}")
        print(f"inference time: {dt}[sec]")
        print()
        if(i>0):
            dt_sum+=dt
    
    dt_avg = dt_sum / (size-1)
    print(f"inference time avg: {dt_avg}[sec]")

if __name__ == "__main__":
    main()
    