import os
import itertools
import gc
import copy
import numpy as np
import tqdm
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import List, Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from warmup_scheduler import GradualWarmupScheduler
from torch.optim import Adam, AdamW, SGD
import torchvision.transforms.functional as TF
from torchvision.transforms import transforms
import torch.nn.functional as F

from diffusers.training_utils import EMAModel
from diffusers.optimization import (
    Union, SchedulerType, Optional,
    Optimizer, TYPE_TO_SCHEDULER_FUNCTION
)

from pilot_train.data.pilot_dataset import PilotDataset
from pilot_train.training.logger import Logger, LoggingManager
from pilot_models.model_registry import get_policy_model
from pilot_utils.data.data_utils import VISUALIZATION_IMAGE_SIZE
from pilot_utils.utils import get_delta, get_goal_mask_tensor, get_modal_dropout_mask, deltas_to_actions, mask_target_context
from pilot_utils.train.train_utils import compute_losses, compute_noise_losses
from pilot_models.policy.pidiff import DiffuserScheduler
from pilot_utils.transforms import ObservationTransform



class BasicTrainer:
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
            device: Computation device (CPU or GPU).
            training_cfg (DictConfig): Training-specific configurations.
            data_cfg (DictConfig): Data handling configurations.
            log_cfg (DictConfig): Logging and visualization configurations.
            datasets_cfg (DictConfig): Dataset configurations.
        """

        self.model = model
        self.model_name = self.model.module.name if hasattr(self.model, "module") else self.model.name
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.test_dataloaders = test_dataloaders
        self.device = device

        # Log config
        project_log_folder = os.path.join(log_cfg.project_folder,log_cfg.run_name)
        if not os.path.exists(project_log_folder):
            os.makedirs(project_log_folder)

        self.project_log_folder=project_log_folder
        self.print_log_freq=log_cfg.print_log_freq
        self.eval_log_freq=log_cfg.eval_log_freq
        self.wandb_log_freq = log_cfg.wandb.run.log_freq
        self.image_log_freq=log_cfg.image_log_freq
        self.num_images_log=log_cfg.num_images_log
        self.use_wandb=log_cfg.wandb.run.enable
        self.eval_fraction=log_cfg.eval_fraction
        self.save_model_freq = log_cfg.save_model_freq 
        
        # Training and Model config
        self.current_epoch=training_cfg.current_epoch
        self.epochs=training_cfg.epochs

        # Data config
        self.goal_condition = data_cfg.goal_condition
        self.target_obs_enable = data_cfg.target_observation_enable # TODO: refactoring
        self.normalized=data_cfg.normalize
        self.learn_angle=data_cfg.learn_angle
        self.action_horizon = data_cfg.action_horizon
        self.pred_horizon=data_cfg.pred_horizon
        self.action_context_size= data_cfg.action_context_size

        self.latest_path = os.path.join(self.project_log_folder, f"latest.pth")

        self.best_loss = float('inf')
        
        self.logging_manager = LoggingManager(datasets_cfg=datasets_cfg,log_cfg=log_cfg)

        self.use_ema = training_cfg.use_ema

        self.ema_model = None
        if self.use_ema: 
            print("Using EMA model")
            self.ema_model = copy.deepcopy(self.model)
            self.ema_model.to(self.device)
            self.ema = hydra.utils.instantiate(
                training_cfg.ema,
                parameters=self.ema_model.parameters())
            # self.ema = EMAModel(self.ema_model.parameters(), power=0.75)
                

        self.goal_mask_prob = training_cfg.goal_mask_prob
        self.goal_mask_prob = torch.clip(torch.tensor(self.goal_mask_prob), 0, 1)

        self.modal_dropout_prob = training_cfg.modal_dropout_prob
        self.modal_dropout_prob = torch.clip(torch.tensor(self.modal_dropout_prob), 0, 1)

        self.debug = training_cfg.debug
        self.global_step = 0
        self.gradient_accumulate_every = training_cfg.gradient_accumulate_every

    def train_one_epoch(self, epoch: int)->None:
        """
        Conducts one epoch of training on the provided data loader and updates the model's parameters.

        Parameters:
            epoch (int): The current epoch number to use for logging and progress tracking.

        Returns:
            None
        """
        raise NotImplementedError("Subclasses must implement this method.")

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
        
        raise NotImplementedError("Subclasses must implement this method.")

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
                f"Start {self.model_name} Model Training Epoch {epoch}/{self.current_epoch + self.epochs - 1}"
            )
            
            self.train_one_epoch(epoch=epoch)
            
            if self.use_ema:
                self.ema.copy_to(self.ema_model.parameters())
            
            avg_total_test_loss = []
            for dataset_type in self.test_dataloaders:
                print(
                    f"Start {dataset_type} {self.model_name} Testing Epoch {epoch}/{self.current_epoch + self.epochs - 1}"
                )
                loader = self.test_dataloaders[dataset_type]
                
                test_action_loss, total_eval_loss = self.evaluate_one_epoch(eval_type=dataset_type,
                                                                                            dataloader=loader,
                                                                                            epoch=epoch)
                avg_total_test_loss.append(total_eval_loss)

            # for all the dataset_type
            current_avg_loss = np.mean(avg_total_test_loss)
            
            print("\033[33m" +f"Best Test loss: {self.best_loss} | Current epoch Test loss: {current_avg_loss}"+ "\033[0m")
            # Update best loss and save the model if current average loss is the new best
            
            if current_avg_loss < self.best_loss:
                best_model_path = os.path.join(self.project_log_folder, "best_model.pth")
                print("\033[32m" + f"Test loss {self.best_loss} decreasing >> {current_avg_loss}\nSaving best model to {best_model_path}" + "\033[0m")
                self.best_loss = current_avg_loss
                self.save_checkpoint(epoch=epoch,
                                    current_avg_loss=current_avg_loss,
                                    path=best_model_path)

            wandb.log({}, commit=False)

            wandb.log({
                "avg_total_test_loss": current_avg_loss,
                "lr": self.optimizer.param_groups[0]["lr"],
            }, commit=False)

            numbered_path = os.path.join(self.project_log_folder, f"{epoch}.pth")
            
            if epoch % self.save_model_freq == 0:
                self.save_checkpoint(epoch=epoch,
                                    current_avg_loss=current_avg_loss,
                                    path=numbered_path)

            # keep track of model at every epoch
            self.save_checkpoint(epoch=epoch,
                                    current_avg_loss=current_avg_loss,
                                    path=self.latest_path)
        # Flush the last set of eval logs
        wandb.log({})
        print()
    
    def save_checkpoint(self,epoch, current_avg_loss, path):
        # DataParallel wrappers keep raw model object in .module attribute
        if self.use_ema:
            raw_model = self.ema_model.module if hasattr(self.ema_model, "module") else self.ema_model
        else:
            raw_model = self.model.module if hasattr(self.model, "module") else self.model
        checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                    "avg_total_test_loss": current_avg_loss
                }
        torch.save(checkpoint, path)
        
    @staticmethod
    def get_model(policy_model_cfg:DictConfig,
                vision_encoder_model_cfg:DictConfig ,
                linear_encoder_model_cfg:DictConfig,
                data_cfg:DictConfig)->nn.Module:
        """
        Constructs and returns a model based on the provided configurations.

        Parameters:
            policy_model_cfg (DictConfig): Configuration for the policy part of the model.
            vision_encoder_model_cfg (DictConfig): Configuration for the encoder part of the model.
            training_cfg (DictConfig): Training configurations.
            data_cfg (DictConfig): Data configurations.

        Returns:
            nn.Module: The constructed model.
        """
    
        return get_policy_model(
                policy_model_cfg = policy_model_cfg,
                vision_encoder_model_cfg = vision_encoder_model_cfg,
                linear_encoder_model_cfg = linear_encoder_model_cfg,
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
    def get_scheduler(
        name: Union[str, SchedulerType],
        optimizer: Optimizer,
        num_warmup_steps: Optional[int] = None,
        num_training_steps: Optional[int] = None,
        **kwargs
    ):
        """
        Added kwargs vs diffuser's original implementation

        Unified API to get any scheduler from its name.

        Args:
            name (`str` or `SchedulerType`):
                The name of the scheduler to use.
            optimizer (`torch.optim.Optimizer`):
                The optimizer that will be used during training.
            num_warmup_steps (`int`, *optional*):
                The number of warmup steps to do. This is not required by all schedulers (hence the argument being
                optional), the function will raise an error if it's unset and the scheduler type requires it.
            num_training_steps (`int``, *optional*):
                The number of training steps to do. This is not required by all schedulers (hence the argument being
                optional), the function will raise an error if it's unset and the scheduler type requires it.
        """
        name = SchedulerType(name)
        schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
        if name == SchedulerType.CONSTANT:
            return schedule_func(optimizer, **kwargs)

        # All other schedulers require `num_warmup_steps`
        if num_warmup_steps is None:
            raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

        if name == SchedulerType.CONSTANT_WITH_WARMUP:
            return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **kwargs)

        # All other schedulers require `num_training_steps`
        if num_training_steps is None:
            raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, **kwargs)
    
    @staticmethod
    def get_dataloaders(datasets_cfg:DictConfig,
                        data_cfg:DictConfig,
                        training_cfg:DictConfig,
                        transform: ObservationTransform)->Tuple:
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
                            data_split_type=data_split_type,
                            transform=transform.get_transform(data_split_type)
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