import numpy as np
import torch
import tqdm
import wandb
import os
from pilot_utils.visualizing import Visualizer
from pilot_utils.utils import to_numpy

class Logger:
    def __init__(
        self,
        name: str,
        dataset: str,
        window_size: int = 10,
        rounding: int = 4,
    ):
        """
        Args:
            name (str): Name of the metric
            dataset (str): Name of the dataset
            window_size (int, optional): Size of the moving average window. Defaults to 10.
            rounding (int, optional): Number of decimals to round to. Defaults to 4.
        """
        self.data = []
        self.name = name
        self.dataset = dataset
        self.rounding = rounding
        self.window_size = window_size

    def display(self) -> str:
        latest = round(self.latest(), self.rounding)
        average = round(self.average(), self.rounding)
        moving_average = round(self.moving_average(), self.rounding)
        output = f"{self.full_name()}: {latest} ({self.window_size}pt moving_avg: {moving_average}) (avg: {average})"
        return output

    def log_data(self, data: float):
        if not np.isnan(data):
            self.data.append(data)

    def full_name(self) -> str:
        return f"{self.name} ({self.dataset})"

    def latest(self) -> float:
        if len(self.data) > 0:
            return self.data[-1]
        return np.nan

    def average(self) -> float:
        if len(self.data) > 0:
            return np.mean(self.data)
        return np.nan

    def moving_average(self) -> float:
        if len(self.data) > self.window_size:
            return np.mean(self.data[-self.window_size :])
        return self.average()

class LoggingManager:
    
    def __init__(self, datasets_cfg, log_cfg) -> None:
        
        log_folder_path = os.path.join(log_cfg.project_folder,
                                    log_cfg.run_name)
        self.log_folder = log_folder_path
        self.dataset_config = datasets_cfg
        self.use_wandb = log_cfg.wandb.run.enable
        self.wandb_log_freq=log_cfg.wandb.run.log_freq
        self.print_log_freq=log_cfg.print_log_freq
        self.eval_log_freq = log_cfg.eval_log_freq
        self.image_log_freq=log_cfg.image_log_freq
        self.num_images_log = log_cfg.num_images_log
        self.visualizer = Visualizer(datasets_cfg=datasets_cfg,
                                    log_cfg=log_cfg)

    def log_data(self,
            i,
            epoch,
            num_batches,
            normalized,
            loggers,
            obs_image,
            goal_image,
            action_pred,
            action_label,
            action_context,
            action_mask,
            goal_pos,
            dataset_index,
            mode,
            use_latest,
            wandb_increment_step=True,
    ):
        """
        Log data to wandb and print to console.
        """
        data_log = {}
        
        if mode == 'train':
            freq_log = self.print_log_freq
        else:
            freq_log = self.eval_log_freq
            
        for key, logger in loggers.items():
            if use_latest:
                data_log[logger.full_name()] = logger.latest()
                if i % freq_log == 0 and freq_log != 0:
                    print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")
            else:
                data_log[logger.full_name()] = logger.average()
                if i % freq_log == 0 and freq_log != 0:
                    print(f"(epoch {epoch}) {logger.full_name()} {logger.average()}")

        if self.use_wandb and i % self.wandb_log_freq == 0 and self.wandb_log_freq != 0:
            wandb.log(data_log, commit=wandb_increment_step)

        if i !=0 and self.image_log_freq != 0 and  i % (self.image_log_freq*self.print_log_freq) == 0:
            print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) Visualize Figures ...")
            
            self.visualizer.visualize_traj_pred(
                to_numpy(obs_image),
                to_numpy(goal_image),
                to_numpy(dataset_index),
                to_numpy(goal_pos),
                to_numpy(action_pred),
                to_numpy(action_label),
                to_numpy(action_context),
                to_numpy(action_mask),
                mode,
                normalized,
                epoch,
            )