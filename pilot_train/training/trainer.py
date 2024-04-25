import os
import json
import itertools

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from warmup_scheduler import GradualWarmupScheduler
from torch.optim import Adam, AdamW, SGD
import torchvision.transforms.functional as TF
from pilot_train.data.data_utils import VISUALIZATION_IMAGE_SIZE
from torchvision import transforms
from omegaconf import DictConfig, OmegaConf


import torch
import numpy as np
import tqdm
import wandb


from pilot_train.training.logger import Logger
from pilot_train.training.train_utils import _compute_losses, _log_data
from pilot_models.policy.model_registry import get_policy_model
from pilot_train.data.pilot_dataset import PilotDataset
from torch.utils.data import DataLoader, ConcatDataset

class Trainer:
    def __init__(self, model, optimizer, scheduler, dataloader,test_dataloaders, transform, device, training_cfg, data_cfg, log_cfg):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.test_dataloaders = test_dataloaders
        self.transform = transform
        self.device = device
        self.training_cfg = training_cfg
        self.data_cfg = data_cfg
        self.log_cfg = log_cfg

        # Log config
        self.project_folder=log_cfg.project_folder
        self.print_log_freq=log_cfg.print_log_freq
        self.wandb_log_freq = log_cfg.wandb.run.log_freq
        self.image_log_freq=log_cfg.image_log_freq
        self.num_images_log=log_cfg.num_images_log
        self.use_wandb=log_cfg.wandb.run.enable
        self.eval_fraction=log_cfg.eval_fraction
        
        # Training config
        self.alpha=training_cfg.alpha if training_cfg.goal_condition else 0
        self.beta=training_cfg.beta
        self.goal_condition = training_cfg.goal_condition
        self.current_epoch=training_cfg.current_epoch
        self.epochs=training_cfg.epochs
        
        # Data config
        self.normalized=data_cfg.normalize
        self.learn_angle=data_cfg.learn_angle
        
        assert 0 <= self.alpha <= 1
        self.latest_path = os.path.join(self.project_folder, f"latest.pth") # TODO: change

        self.best_loss = float('inf')

    def train_one_epoch(self, epoch):

        self.model.train()
        dist_loss_logger = Logger("dist_loss", "train", window_size=self.print_log_freq)
        action_loss_logger = Logger("action_loss", "train", window_size=self.print_log_freq)
        action_waypts_cos_sim_logger = Logger(
            "action_waypts_cos_sim", "train", window_size=self.print_log_freq
        )
        multi_action_waypts_cos_sim_logger = Logger(
            "multi_action_waypts_cos_sim", "train", window_size=self.print_log_freq
        )
        total_loss_logger = Logger("total_loss", "train", window_size=self.print_log_freq)
        loggers = {
            "dist_loss": dist_loss_logger,
            "action_loss": action_loss_logger,
            "action_waypts_cos_sim": action_waypts_cos_sim_logger,
            "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim_logger,
            "total_loss": total_loss_logger,
        }

        if self.learn_angle:
            action_orien_cos_sim_logger = Logger(
                "action_orien_cos_sim", "train", window_size=self.print_log_freq
            )
            multi_action_orien_cos_sim_logger = Logger(
                "multi_action_orien_cos_sim", "train", window_size=self.print_log_freq
            )
            loggers["action_orien_cos_sim"] = action_orien_cos_sim_logger
            loggers["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim_logger

        num_batches = len(self.dataloader)
        tqdm_iter = tqdm.tqdm(
            self.dataloader,
            dynamic_ncols=True,
            desc=f"Training epoch {epoch}",
        )
        for i, data in enumerate(tqdm_iter):
            (
                obs_image,
                goal_image,
                action_label,
                dist_label,
                goal_pos,
                dataset_index,
                action_mask,
            ) = data

            obs_images = torch.split(obs_image, 3, dim=1)
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE)
            obs_images = [self.transform(obs_image).to(self.device) for obs_image in obs_images]
            obs_image = torch.cat(obs_images, dim=1)

            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE)

            
            #### TODO: change the goal image to relative position to target
            # will be taken from the __getitem__
            goal_image = self.transform(goal_image).to(self.device)
            model_outputs = self.model(obs_image, goal_image)

            #### TODO: change the dist label 
            dist_label = dist_label.to(self.device)
            
            action_label = action_label.to(self.device)
            action_mask = action_mask.to(self.device)

            self.optimizer.zero_grad()

            dist_pred, action_pred = model_outputs

            losses = _compute_losses(
                dist_label=dist_label,
                action_label=action_label,
                dist_pred=dist_pred,
                action_pred=action_pred,
                alpha= self.alpha,
                beta = self.beta,
                learn_angle=self.learn_angle,
                action_mask=action_mask,
            )

            # Backward step
            losses["total_loss"].backward()
            self.optimizer.step()

            # Appenf to Logger
            for key, value in losses.items():
                if key in loggers:
                    logger = loggers[key]
                    logger.log_data(value.item())

            _log_data(
                i=i,
                epoch=epoch,
                num_batches=num_batches,
                normalized=self.normalized,
                project_folder=self.project_folder,
                num_images_log=self.num_images_log,
                loggers=loggers,
                obs_image=viz_obs_image,
                goal_image=viz_goal_image,
                action_pred=action_pred,
                action_label=action_label,
                dist_pred=dist_pred,
                dist_label=dist_label,
                goal_pos=goal_pos,
                dataset_index=dataset_index,
                wandb_log_freq=self.wandb_log_freq,
                print_log_freq=self.print_log_freq,
                image_log_freq=self.image_log_freq,
                use_wandb=self.use_wandb,
                mode="train",
                use_latest=True,
            )

    def evaluate_one_epoch(self, dataloader, eval_type, epoch):
        self.model.eval()
        dist_loss_logger = Logger("dist_loss", eval_type)
        action_loss_logger = Logger("action_loss", eval_type)
        action_waypts_cos_sim_logger = Logger("action_waypts_cos_sim", eval_type)
        multi_action_waypts_cos_sim_logger = Logger("multi_action_waypts_cos_sim", eval_type)
        total_loss_logger = Logger("total_loss", eval_type)
        loggers = {
            "dist_loss": dist_loss_logger,
            "action_loss": action_loss_logger,
            "action_waypts_cos_sim": action_waypts_cos_sim_logger,
            "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim_logger,
            "total_loss": total_loss_logger,
        }

        if self.learn_angle:
            action_orien_cos_sim_logger = Logger("action_orien_cos_sim", eval_type)
            multi_action_orien_cos_sim_logger = Logger("multi_action_orien_cos_sim", eval_type)
            loggers["action_orien_cos_sim"] = action_orien_cos_sim_logger
            loggers["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim_logger

        num_batches = len(dataloader)
        num_batches = max(int(num_batches * self.eval_fraction), 1)

        viz_obs_image = None
        with torch.no_grad():
            tqdm_iter = tqdm.tqdm(
                itertools.islice(dataloader, num_batches),
                total=num_batches,
                dynamic_ncols=True,
                desc=f"Evaluating {eval_type} for epoch {epoch}",
            )
            for i, data in enumerate(tqdm_iter):
                (
                    obs_image,
                    goal_image,
                    action_label,
                    dist_label,
                    goal_pos,
                    dataset_index,
                    action_mask,
                ) = data

                obs_images = torch.split(obs_image, 3, dim=1)
                viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE)
                obs_images = [self.transform(obs_image).to(self.device) for obs_image in obs_images]
                obs_image = torch.cat(obs_images, dim=1)

                viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE)

                goal_image = self.transform(goal_image).to(self.device)
                model_outputs = self.model(obs_image, goal_image)

                dist_label = dist_label.to(self.device)
                action_label = action_label.to(self.device)
                action_mask = action_mask.to(self.device)

                dist_pred, action_pred = model_outputs
                # print()
                # print("GT sample: ")
                # print(action_label[0][0])
                # print("Action predictions sample: ")
                # print(action_pred[0][0])
                # print()
                losses = _compute_losses(
                    dist_label=dist_label,
                    action_label=action_label,
                    dist_pred=dist_pred,
                    action_pred=action_pred,
                    alpha=self.alpha,
                    beta=self.beta,
                    learn_angle=self.learn_angle,
                    action_mask=action_mask,
                )

                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value.item())

        # Log data to wandb/console, with visualizations selected from the last batch
        _log_data(
            i=i,
            epoch=epoch,
            num_batches=num_batches,
            normalized=self.normalized,
            project_folder=self.project_folder,
            num_images_log=self.num_images_log,
            loggers=loggers,
            obs_image=viz_obs_image,
            goal_image=viz_goal_image,
            action_pred=action_pred,
            action_label=action_label,
            goal_pos=goal_pos,
            dist_pred=dist_pred,
            dist_label=dist_label,
            dataset_index=dataset_index,
            use_wandb=self.use_wandb,
            mode=eval_type,
            use_latest=False,
            wandb_increment_step=False,
        )

        return dist_loss_logger.average(), action_loss_logger.average(), total_loss_logger.average()

    def run(self):
        for epoch in range(self.current_epoch, self.current_epoch + self.epochs):
            print()
            print(
                f"Start {self.model.name} Model Training Epoch {epoch}/{self.current_epoch + self.epochs - 1}"
            )
            
            self.train_one_epoch(epoch=epoch)
            
            avg_total_test_loss = []
            for dataset_type in self.test_dataloaders:
                print(
                    f"Start {dataset_type} {self.model.name} Testing Epoch {epoch}/{self.current_epoch + self.epochs - 1}"
                )
                loader = self.test_dataloaders[dataset_type]
                test_dist_loss, test_action_loss, total_eval_loss = self.evaluate_one_epoch(eval_type=dataset_type,
                                                                                            dataloader=loader,
                                                                                            epoch=epoch)
                avg_total_test_loss.append(total_eval_loss)

            current_avg_loss = np.mean(avg_total_test_loss)
            # Update best loss and save the model if current average loss is the new best
            if current_avg_loss < self.best_loss:
                best_model_path = os.path.join(self.project_folder, "best_model.pth")
                print(f"Avg Test loss {self.best_loss} decreasing >> {current_avg_loss}\nSaving best model to {best_model_path}")
                self.best_loss = current_avg_loss
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                    "avg_total_test_loss": current_avg_loss
                }
                torch.save(checkpoint, best_model_path)

            checkpoint = {
                "epoch": epoch,
                "model": self.model,
                "optimizer": self.optimizer,
                "avg_total_test_loss": current_avg_loss,
                "scheduler": self.scheduler
            }
            # log average eval loss
            wandb.log({}, commit=False)

            if self.scheduler is not None:
                # scheduler calls based on the type of scheduler
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(current_avg_loss)
                else:
                    self.scheduler.step()
            wandb.log({
                "avg_total_test_loss": current_avg_loss,
                "lr": self.optimizer.param_groups[0]["lr"],
            }, commit=False)

            numbered_path = os.path.join(self.project_folder, f"{epoch}.pth")
            torch.save(checkpoint, self.latest_path)
            torch.save(checkpoint, numbered_path)  # keep track of model at every epoch

        # Flush the last set of eval logs
        wandb.log({})
        print()
    
    @staticmethod
    def get_model(policy_model_cfg, encoder_model_cfg , training_cfg, data_cfg ):
    
        return get_policy_model(
                policy_model_cfg = policy_model_cfg,
                encoder_model_cfg = encoder_model_cfg,
                training_cfg = training_cfg,
                data_cfg = data_cfg
                )

    @staticmethod
    def get_optimizer(optimizer_name:str, model:nn.Module, lr:float)->torch.optim:
        optimizer_name = optimizer_name.lower()
        if optimizer_name == "adam":
            optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
        elif optimizer_name == "adamw":
            optimizer = AdamW(model.parameters(), lr=lr)
        elif optimizer_name == "sgd":
            optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Optimizer {optimizer_name} not supported")

        return optimizer
    
    @staticmethod
    def get_scheduler(training_cfg:DictConfig, optimizer:torch.optim, lr:float):
        scheduler_name = training_cfg.scheduler.lower()
        if scheduler_name == "cosine":
            print("Using cosine annealing with T_max", training_cfg.epochs)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=training_cfg.epochs
            )
        elif scheduler_name == "cyclic":
            print("Using cyclic LR with cycle", training_cfg.cyclic_period)
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=lr / 10.,
                max_lr=lr,
                step_size_up=training_cfg.cyclic_period // 2,
                cycle_momentum=False,
            )
        elif scheduler_name == "plateau":
            print("Using ReduceLROnPlateau")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=training_cfg.plateau_factor,
                patience=training_cfg.plateau_patience,
                verbose=True,
            )
        else:
            raise ValueError(f"Scheduler {scheduler_name} not supported")

        if training_cfg.warmup:
            print("Using warmup scheduler")
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=training_cfg.warmup_epochs,
                after_scheduler=scheduler,
            )
        
        return scheduler
    
    
    @staticmethod
    def get_dataloaders(datasets_cfg, data_cfg, training_cfg):
        train_dataset = []
        test_dataloaders = {}

        for robot in datasets_cfg.robots:
            robot_dataset_cfg = datasets_cfg[robot]
            if "negative_mining" not in robot_dataset_cfg:
                robot_dataset_cfg["negative_mining"] = True
            if "goals_per_obs" not in robot_dataset_cfg:
                robot_dataset_cfg["goals_per_obs"] = 1
            if "end_slack" not in robot_dataset_cfg:
                robot_dataset_cfg["end_slack"] = 0
            if "waypoint_spacing" not in robot_dataset_cfg:
                OmegaConf.update(robot_dataset_cfg, "waypoint_spacing",1, force_add=True)

            for data_split_type in ["train", "test"]:
                if data_split_type in robot_dataset_cfg:
                        dataset = PilotDataset(
                            data_cfg = data_cfg,
                            datasets_cfg = datasets_cfg,
                            robot_dataset_cfg = robot_dataset_cfg,
                            dataset_name=robot,
                            data_split_type=data_split_type
                        )
                        if data_split_type == "train":
                            train_dataset.append(dataset)
                        else:
                            dataset_type = f"{robot}_{data_split_type}"
                            if dataset_type not in test_dataloaders:
                                test_dataloaders[dataset_type] = {}
                            test_dataloaders[dataset_type] = dataset

        # combine all the datasets from different robots
        train_dataset = ConcatDataset(train_dataset)

        train_loader = DataLoader(
            train_dataset,
            batch_size=training_cfg.batch_size,
            shuffle=True,
            num_workers=training_cfg.num_workers,
            drop_last=False,
            persistent_workers=True,
        )

        for dataset_type, dataset in test_dataloaders.items():
            test_dataloaders[dataset_type] = DataLoader(
                dataset,
                batch_size=training_cfg.eval_batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=False,
            )
        
        return train_loader, test_dataloaders