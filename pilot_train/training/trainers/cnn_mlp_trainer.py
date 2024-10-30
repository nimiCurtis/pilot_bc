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
from pilot_train.training.trainers.basic_trainer import BasicTrainer



class CNNMLPTrainer(BasicTrainer):
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

        super().__init__(model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        dataloader=dataloader,
                        test_dataloaders=test_dataloaders,
                        device=device,
                        training_cfg=training_cfg,
                        data_cfg=data_cfg, 
                        log_cfg=log_cfg,
                        datasets_cfg=datasets_cfg)

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

        # Generate random goal mask
        
        num_batches = len(self.dataloader)
        tqdm_iter = tqdm.tqdm(
            self.dataloader,
            dynamic_ncols=True,
            desc=f"Training epoch {epoch}",
        )
        total_train_steps = tqdm_iter.total
        
        for i, data in enumerate(tqdm_iter):
            (
                vision_obs_context,
                rel_pos_to_target_context,
                goal_rel_pos_to_target,
                normalized_actions,
                normalized_actions_context,
                normalized_goal_pos,
                dataset_index,
                action_mask,
                goal_pos_to_target_mask
            ) = data

            action_dim = normalized_actions.shape[-1]
            # STATE
            # visual context ###TODO: refactore
            viz_images = torch.split(vision_obs_context, 1, dim=1)
            viz_obs_image = TF.resize(viz_images[-1], VISUALIZATION_IMAGE_SIZE)
            viz_context_t0_image = TF.resize(viz_images[max(-self.action_context_size,-self.context_size)], VISUALIZATION_IMAGE_SIZE)

            vision_obs_context = vision_obs_context.to(self.device)
            
            # TARGET
            rel_pos_to_target_context = rel_pos_to_target_context.to(self.device)

            # ACTION
            action_label_pred = normalized_actions.to(self.device)
            normalized_actions_context = normalized_actions_context.to(self.device)
            # action label pred -> the pred horizon actions. it is already normalize cumsum
            # so the deltas are normalized
            
            # Take the actions horizon samples for the action loss
            action_label = normalized_actions[:,:self.action_horizon,:]
            action_label = action_label.to(self.device)
            action_mask = action_mask.to(self.device)
            
            # Take action label deltas
            action_label_pred_deltas = get_delta(actions=action_label_pred[:,:,:2]) # deltas of x,y
            
            if self.learn_angle:
                # We don't take the deltas of yaw
                action_label_pred_deltas = torch.cat([action_label_pred_deltas, action_label_pred[:,:,2:]], dim=2)

            # Predict the noise residual
            obs_encoding_condition = self.model("vision_encoder",obs_img=vision_obs_context)

            linear_input = torch.concatenate([rel_pos_to_target_context.flatten(1),
                                            normalized_actions_context.flatten(1)], axis=1)

            lin_encoding = self.model("linear_encoder",
                                    curr_rel_pos_to_target=linear_input)
                            
            modalities = [obs_encoding_condition, lin_encoding]
            
            
            # Not in use!
            modal_dropout_mask = get_modal_dropout_mask(self.train_batch_size,modalities_size=len(modalities),curr_rel_pos_to_target=rel_pos_to_target_context,modal_dropout_prob=self.modal_dropout_prob).to(self.device)   # modify
            
            final_encoded_condition = self.model("fuse_modalities",
                                                modalities=modalities,
                                                mask=modal_dropout_mask)


            cnn_mlp_output = self.model("action_pred",
                                    final_encoded_condition=final_encoded_condition)

            # action_pred = action_pred[:,:self.action_horizon,:]
            action_pred = deltas_to_actions(deltas=cnn_mlp_output,
                                            pred_horizon=self.pred_horizon,
                                            action_horizon=self.action_horizon,
                                            learn_angle=self.learn_angle)
            
            losses = compute_losses(
                action_label=action_label,
                action_pred=action_pred,
                action_mask=action_mask,
            )
            
            loss = losses["total_loss"]

            loss.backward()

            # step optimizer
            if self.global_step % self.gradient_accumulate_every == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                            
            # Update Exponential Moving Average of the model weights after optimizing
            if self.use_ema:
                self.ema.step(self.model.parameters())

            # Append to Logger
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
                goal_image=viz_context_t0_image,
                action_pred=action_pred,
                action_label=action_label,
                action_context=normalized_actions_context,
                action_mask=action_mask,
                goal_pos=normalized_goal_pos,
                dataset_index=dataset_index,
                mode="train",
                use_latest=True,
            )

            is_last_batch = i == total_train_steps - 2
            if self.debug.max_train_steps:
                if i==self.debug.max_train_steps:
                    break
            elif is_last_batch:
                    break
            else:
                self.global_step+=1

        if self.scheduler is not None:
            self.scheduler.step()

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

        eval_model = self.ema_model if self.use_ema else self.model
        eval_model.eval()
        
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
                    vision_obs_context,
                    rel_pos_to_target_context,
                    goal_rel_pos_to_target,
                    normalized_actions,
                    normalized_actions_context,
                    normalized_goal_pos,
                    dataset_index,
                    action_mask,
                    goal_pos_to_target_mask
                ) = data

                action_dim = normalized_actions.shape[-1]
                # STATE
                # visual context ###TODO: refactore
                viz_images = torch.split(vision_obs_context, 1, dim=1)
                viz_obs_image = TF.resize(viz_images[-1], VISUALIZATION_IMAGE_SIZE)
                viz_context_t0_image = TF.resize(viz_images[max(-self.action_context_size,-self.context_size)], VISUALIZATION_IMAGE_SIZE)

                vision_obs_context = vision_obs_context.to(self.device)
                
                # TARGET
                rel_pos_to_target_context = rel_pos_to_target_context.to(self.device)

                # ACTION
                action_label_pred = normalized_actions.to(self.device)
                normalized_actions_context = normalized_actions_context.to(self.device)
                # action label pred -> the pred horizon actions. it is already normalize cumsum
                # so the deltas are normalized
                
                # Take the actions horizon samples for the action loss
                action_label = normalized_actions[:,:self.action_horizon,:]
                action_label = action_label.to(self.device)
                action_mask = action_mask.to(self.device)
                
                # Take action label deltas
                action_label_pred_deltas = get_delta(actions=action_label_pred[:,:,:2]) # deltas of x,y
                
                if self.learn_angle:
                    # We don't take the deltas of yaw
                    action_label_pred_deltas = torch.cat([action_label_pred_deltas, action_label_pred[:,:,2:]], dim=2)

                # Predict the noise residual
                obs_encoding_condition = self.model("vision_encoder",obs_img=vision_obs_context)

                linear_input = torch.concatenate([rel_pos_to_target_context.flatten(1),
                                                normalized_actions_context.flatten(1)], axis=1)

                lin_encoding = self.model("linear_encoder",
                                        curr_rel_pos_to_target=linear_input)
                                
                modalities = [obs_encoding_condition, lin_encoding]
                
                
                # Not in use!
                modal_dropout_mask = get_modal_dropout_mask(self.eval_batch_size,modalities_size=len(modalities),curr_rel_pos_to_target=rel_pos_to_target_context,modal_dropout_prob=self.modal_dropout_prob).to(self.device)   # modify
                
                final_encoded_condition = self.model("fuse_modalities",
                                                    modalities=modalities,
                                                    mask=modal_dropout_mask)


                cnn_mlp_output = self.model("action_pred",
                                        final_encoded_condition=final_encoded_condition)

                # action_pred = action_pred[:,:self.action_horizon,:]
                action_pred = deltas_to_actions(deltas=cnn_mlp_output,
                                                pred_horizon=self.pred_horizon,
                                                action_horizon=self.action_horizon,
                                                learn_angle=self.learn_angle)
                
                losses = compute_losses(
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
                    goal_image=viz_context_t0_image,
                    action_pred=action_pred,
                    action_label=action_label,
                    action_context=normalized_actions_context,
                    action_mask=action_mask,
                    goal_pos=normalized_goal_pos,
                    dataset_index=dataset_index,
                    mode=eval_type,
                    use_latest=False,
                    wandb_increment_step=False,
                )
                
                if self.debug.max_val_steps:
                    if i==self.debug.max_val_steps:
                        break

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

            ## TODO:check
            # if self.scheduler is not None:
            #     # scheduler calls based on the type of scheduler
            #     if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            #         self.scheduler.step(current_avg_loss)
            #     else:
            #         self.scheduler.step()
            
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
    
    # @staticmethod
    # def get_scheduler(training_cfg:DictConfig, optimizer:torch.optim, lr:float):
    #     """
    #     Constructs and returns a learning rate scheduler based on the provided configuration.

    #     Parameters:
    #         training_cfg (DictConfig): Training configurations that may include scheduler type.
    #         optimizer (torch.optim): Optimizer for which the scheduler will be used.
    #         lr (float): Initial learning rate.

    #     Returns:
    #         Scheduler: The constructed scheduler.
    #     """
        
    #     scheduler_name = training_cfg.scheduler.lower()
    #     if scheduler_name == "cosine":
    #         print("Using cosine annealing with T_max", training_cfg.epochs)
    #         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #             optimizer, T_max=training_cfg.epochs
    #         )
    #     elif scheduler_name == "cyclic":
    #         print("Using cyclic LR with cycle", training_cfg.cyclic_period)
    #         scheduler = torch.optim.lr_scheduler.CyclicLR(
    #             optimizer,
    #             base_lr=lr / 10.,
    #             max_lr=lr,
    #             step_size_up=training_cfg.cyclic_period // 2,
    #             cycle_momentum=False,
    #         )
    #     elif scheduler_name == "plateau":
    #         print("Using ReduceLROnPlateau")
    #         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #             optimizer,
    #             factor=training_cfg.plateau_factor,
    #             patience=training_cfg.plateau_patience,
    #             verbose=True,
    #         )
    #     else:
    #         raise ValueError(f"Scheduler {scheduler_name} not supported")

    #     if training_cfg.warmup:
    #         print("Using warmup scheduler")
    #         scheduler = GradualWarmupScheduler(
    #             optimizer,
    #             multiplier=1,
    #             total_epoch=training_cfg.warmup_epochs,
    #             after_scheduler=scheduler,
    #         )
        
    #     return scheduler
    
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