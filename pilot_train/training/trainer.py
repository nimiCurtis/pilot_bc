import os
import itertools

import numpy as np
import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf
from typing import List, Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from warmup_scheduler import GradualWarmupScheduler
from torch.optim import Adam, AdamW, SGD
import torchvision.transforms.functional as TF
from torchvision.transforms import transforms

from pilot_train.data.pilot_dataset import PilotDataset
from pilot_train.training.logger import Logger, LoggingManager
from pilot_models.policy.model_registry import get_policy_model
from pilot_utils.data.data_utils import VISUALIZATION_IMAGE_SIZE

class Trainer:
    """
    A class responsible for managing the training and evaluation processes of pilot model.

    This class handles the operations necessary to train a model, evaluate it on test data, log results,
    and save model checkpoints using a set of configurations provided during instantiation.

    """
        
    def __init__(self, model: nn.Module,
                optimizer: torch.optim,
                scheduler,
                dataloader: DataLoader,
                test_dataloaders: List[DataLoader],
                transform: transforms ,
                device,
                training_cfg: DictConfig,
                data_cfg: DictConfig, 
                log_cfg: DictConfig,
                datasets_cfg: DictConfig):
        """
        Initializes the Trainer object with model, optimizer, scheduler, data loaders, and configurations.

        Parameters:
            model (nn.Module): The model to be trained.
            optimizer (torch.optim): Optimizer for training the model.
            scheduler: Learning rate scheduler.
            dataloader (DataLoader): Loader for training data.
            test_dataloaders (List[DataLoader]): List of loaders for test data.
            transform (transforms): Data transformations.
            device: Computation device (CPU or GPU).
            training_cfg (DictConfig): Training-specific configurations.
            data_cfg (DictConfig): Data handling configurations.
            log_cfg (DictConfig): Logging and visualization configurations.
            datasets_cfg (DictConfig): Dataset configurations.
        """
        
        
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
        project_log_folder = os.path.join(log_cfg.project_folder,log_cfg.run_name)
        if not os.path.exists(project_log_folder):
            os.makedirs(project_log_folder)

        self.project_log_folder=project_log_folder
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
        self.latest_path = os.path.join(self.project_log_folder, f"latest.pth") # TODO: change

        self.best_loss = float('inf')
        
        self.logging_manager = LoggingManager(datasets_cfg=datasets_cfg,log_cfg=log_cfg)

    def train_one_epoch(self, epoch: int)->None:
        """
        Conducts one epoch of training on the provided data loader and updates the model's parameters.

        Parameters:
            epoch (int): The current epoch number to use for logging and progress tracking.

        Returns:
            None
        """
        
        self.model.train()
        action_loss_logger = Logger("action_loss", "train", window_size=self.print_log_freq)
        action_waypts_cos_sim_logger = Logger(
            "action_waypts_cos_sim", "train", window_size=self.print_log_freq
        )
        multi_action_waypts_cos_sim_logger = Logger(
            "multi_action_waypts_cos_sim", "train", window_size=self.print_log_freq
        )
        total_loss_logger = Logger("total_loss", "train", window_size=self.print_log_freq)
        loggers = {
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
                curr_rel_pos_to_target,
                goal_rel_pos_to_target,
                action_label,
                goal_pos,
                dataset_index,
                action_mask,
            ) = data

            # STATE
            # visual context
            obs_images = torch.split(obs_image, 3, dim=1)
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE)
            obs_images = [self.transform(obs_image).to(self.device) for obs_image in obs_images]
            obs_image = torch.cat(obs_images, dim=1)
            # current relative target pos
            curr_rel_pos_to_target = curr_rel_pos_to_target.to(self.device)

            # GOAL
            goal_rel_pos_to_target = goal_rel_pos_to_target.to(self.device)

            # This line is for not corrupt the pipeline of visualization right now
            # TODO: modify it
            viz_goal_image = viz_obs_image

            # Not in use
            # TODO: remove the transform from the script and from the train.py script
            #goal_image = self.transform(goal_image).to(self.device)
            
            # ACTION
            action_label = action_label.to(self.device)
            action_mask = action_mask.to(self.device)
            
            # Infer model
            model_outputs = self.model(obs_image, curr_rel_pos_to_target, goal_rel_pos_to_target)
            action_pred = model_outputs

            self.optimizer.zero_grad()

            losses = self.model._compute_losses(
                action_label=action_label,
                action_pred=action_pred,
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

            self.logging_manager.log_data(
                i=i,
                epoch=epoch,
                num_batches=num_batches,
                normalized=self.normalized,
                loggers=loggers,
                obs_image=viz_obs_image,
                goal_image=viz_goal_image,
                action_pred=action_pred,
                action_label=action_label,
                goal_pos=goal_pos,
                dataset_index=dataset_index,
                mode="train",
                use_latest=True,
            )

    def evaluate_one_epoch(self, dataloader:DataLoader, eval_type:str, epoch:int)->Tuple:
        """
        Evaluates the model's performance on a given dataset for one epoch.

        Parameters:
            dataloader (DataLoader): The DataLoader for the dataset to evaluate.
            eval_type (str): Type of evaluation (e.g., "validation", "test").
            epoch (int): Current epoch number for logging purposes.

        Returns:
            tuple: Tuple containing the average distance loss, action loss, and total loss for the evaluated epoch.
        """
        
        self.model.eval()
        action_loss_logger = Logger("action_loss", eval_type)
        action_waypts_cos_sim_logger = Logger("action_waypts_cos_sim", eval_type)
        multi_action_waypts_cos_sim_logger = Logger("multi_action_waypts_cos_sim", eval_type)
        total_loss_logger = Logger("total_loss", eval_type)
        loggers = {
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
                    curr_rel_pos_to_target,
                    goal_rel_pos_to_target,
                    action_label,
                    goal_pos,
                    dataset_index,
                    action_mask,
                ) = data

                # STATE
                # visual context
                obs_images = torch.split(obs_image, 3, dim=1)
                viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE)
                obs_images = [self.transform(obs_image).to(self.device) for obs_image in obs_images]
                obs_image = torch.cat(obs_images, dim=1)
                # current relative target pos
                curr_rel_pos_to_target = curr_rel_pos_to_target.to(self.device)

                # GOAL
                goal_rel_pos_to_target = goal_rel_pos_to_target.to(self.device)

                # This line is for not corrupt the pipeline of visualization right now
                # TODO: modify it
                viz_goal_image = viz_obs_image

                # Not in use
                # TODO: remove the transform from the script and from the train.py script
                #goal_image = self.transform(goal_image).to(self.device)
                
                # ACTION
                action_label = action_label.to(self.device)
                action_mask = action_mask.to(self.device)
                
                # Infer model
                model_outputs = self.model(obs_image, curr_rel_pos_to_target, goal_rel_pos_to_target)
                action_pred = model_outputs

                losses = self.model._compute_losses(
                    action_label=action_label,
                    action_pred=action_pred,
                    action_mask=action_mask,
                )

                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value.item())

                # Log data to wandb/console, with visualizations selected from the last batch
                self.logging_manager.log_data(
                    i=i,
                    epoch=epoch,
                    num_batches=num_batches,
                    normalized=self.normalized,
                    loggers=loggers,
                    obs_image=viz_obs_image,
                    goal_image=viz_goal_image,
                    action_pred=action_pred,
                    action_label=action_label,
                    goal_pos=goal_pos,
                    dataset_index=dataset_index,
                    mode=eval_type,
                    use_latest=False,
                    wandb_increment_step=False,
                )

        return action_loss_logger.average(), total_loss_logger.average()

    def run(self)->None:
        """
        Executes the training process across all specified epochs, evaluates on test datasets, and manages
        logging and model saving.

        Returns:
            None
        """
        
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
                test_action_loss, total_eval_loss = self.evaluate_one_epoch(eval_type=dataset_type,
                                                                                            dataloader=loader,
                                                                                            epoch=epoch)
                avg_total_test_loss.append(total_eval_loss)

            current_avg_loss = np.mean(avg_total_test_loss)
            
            # Update best loss and save the model if current average loss is the new best
            if current_avg_loss < self.best_loss:
                best_model_path = os.path.join(self.project_log_folder, "best_model.pth")
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

            numbered_path = os.path.join(self.project_log_folder, f"{epoch}.pth")
            torch.save(checkpoint, self.latest_path)
            torch.save(checkpoint, numbered_path)  # keep track of model at every epoch

        # Flush the last set of eval logs
        wandb.log({})
        print()
    
    @staticmethod
    def get_model(policy_model_cfg:DictConfig,
                encoder_model_cfg:DictConfig ,
                training_cfg:DictConfig,
                data_cfg:DictConfig)->nn.Module:
        """
        Constructs and returns a model based on the provided configurations.

        Parameters:
            policy_model_cfg (DictConfig): Configuration for the policy part of the model.
            encoder_model_cfg (DictConfig): Configuration for the encoder part of the model.
            training_cfg (DictConfig): Training configurations.
            data_cfg (DictConfig): Data configurations.

        Returns:
            nn.Module: The constructed model.
        """
    
        return get_policy_model(
                policy_model_cfg = policy_model_cfg,
                encoder_model_cfg = encoder_model_cfg,
                training_cfg = training_cfg,
                data_cfg = data_cfg
                )

    @staticmethod
    def get_optimizer(optimizer_name:str, model:nn.Module, lr:float)->torch.optim:
        """
        Retrieves the optimizer based on the provided name and configuration.

        Parameters:
            optimizer_name (str): Name of the optimizer.
            model (nn.Module): The model for which the optimizer will be used.
            lr (float): Learning rate for the optimizer.

        Returns:
            torch.optim: The constructed optimizer.
        """
        
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
        """
        Constructs and returns a learning rate scheduler based on the provided configuration.

        Parameters:
            training_cfg (DictConfig): Training configurations that may include scheduler type.
            optimizer (torch.optim): Optimizer for which the scheduler will be used.
            lr (float): Initial learning rate.

        Returns:
            Scheduler: The constructed scheduler.
        """
        
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
    def get_dataloaders(datasets_cfg:DictConfig,
                        data_cfg:DictConfig,
                        training_cfg:DictConfig)->Tuple:
        """
        Constructs and returns DataLoaders for training and testing based on provided configurations.

        Parameters:
            datasets_cfg (Config): Configuration defining the datasets to use and their parameters.
            data_cfg (DictConfig): Data-specific configurations.
            training_cfg (DictConfig): Training-specific configurations.

        Returns:
            tuple: A tuple containing the DataLoader for training and a dictionary of DataLoaders for testing.
        """
        
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


### TODO: check for what we need this ??

# def load_model(model, model_type, checkpoint: dict) -> None:
#     """Load model from checkpoint."""
#     if model_type == "nomad":
#         state_dict = checkpoint
#         model.load_state_dict(state_dict, strict=False)
#     else:
#         loaded_model = checkpoint["model"]
#         try:
#             state_dict = loaded_model.module.state_dict()
#             model.load_state_dict(state_dict, strict=False)
#         except AttributeError as e:
#             state_dict = loaded_model.state_dict()
#             model.load_state_dict(state_dict, strict=False)


# def load_ema_model(ema_model, state_dict: dict) -> None:
#     """Load model from checkpoint."""
#     ema_model.load_state_dict(state_dict)