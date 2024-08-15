import os
import itertools
import numpy as np
import tqdm
import wandb
import hydra
from omegaconf import DictConfig
from typing import List, Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

from pilot_train.data.pilot_dataset import PilotDataset
from pilot_train.training.logger import Logger, LoggingManager
from pilot_utils.data.data_utils import VISUALIZATION_IMAGE_SIZE
from pilot_utils.utils import get_delta, get_goal_mask_tensor, get_modal_dropout_mask, deltas_to_actions
from pilot_utils.train.train_utils import compute_losses, compute_noise_losses
from pilot_models.policy.pidiff import DiffuserScheduler
from pilot_train.training.trainers.basic_trainer import BasicTrainer


class PiDiffTrainer(BasicTrainer):
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
        
        noise_scheduler_config = self.model.module.get_scheduler_config() if hasattr(self.model, "module") else self.model.get_scheduler_config()
        self.noise_scheduler = DiffuserScheduler(noise_scheduler_config)


    def train_one_epoch(self, epoch: int)->None:
        """
        Conducts one epoch of training on the provided data loader and updates the model's parameters.

        Parameters:
            epoch (int): The current epoch number to use for logging and progress tracking.

        Returns:
            None
        """
        
        self.model.train()
        self.noise_scheduler.train()
        
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

        if self.model_name == "pidiff":
            diffusion_noise_loss = Logger(
                "diffusion_noise_loss", "train", window_size=self.print_log_freq
            )
            loggers["diffusion_noise_loss"] = diffusion_noise_loss


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
                obs_image_transformed,
                curr_rel_pos_to_target,
                goal_rel_pos_to_target,
                action_label,
                normalized_prev_actions,
                goal_pos,
                dataset_index,
                action_mask,
            ) = data

            B = action_label.shape[0] #batch size
            action_dim = action_label.shape[-1]
            # STATE
            # visual context ###TODO: modify for rgb
            viz_images = torch.split(obs_image_transformed, 1, dim=1)
            viz_obs_image = TF.resize(viz_images[-1], VISUALIZATION_IMAGE_SIZE)
            viz_context_t0_image = TF.resize(viz_images[-self.action_context_size], VISUALIZATION_IMAGE_SIZE)

            obs_image_transformed = obs_image_transformed.to(self.device)
            
            # current relative target pos
            curr_rel_pos_to_target = curr_rel_pos_to_target.to(self.device)

            # GOAL
            goal_rel_pos_to_target = goal_rel_pos_to_target.to(self.device)

            # This line is for not corrupt the pipeline of visualization right now
            # TODO: modify it
            
            # ACTION
            action_label_pred = action_label.to(self.device)
            normalized_prev_actions = normalized_prev_actions.to(self.device)
            # action label pred -> the pred horizon actions. it is already normalize cumsum
            # so the deltas are normalized
            
            # Take the actions horizon samples for the action loss
            action_label = action_label[:,:self.action_horizon,:]
            action_label = action_label.to(self.device)
            action_mask = action_mask.to(self.device)
            
            # Infer model
            if self.model_name == "pidiff":
                
                # Take action label deltas
                action_label_pred_deltas = get_delta(actions=action_label_pred[:,:,:2]) # deltas of x,y,cos_yaw, sin_yaw
                
                if self.learn_angle:
                    # We don't take the deltas of yaw
                    action_label_pred_deltas = torch.cat([action_label_pred_deltas, action_label_pred[:,:,2:]], dim=2)

                # Sample noise to add to actions
                noise = torch.randn(action_label_pred_deltas.shape, device=self.device)

                # Sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0, self.noise_scheduler.noise_scheduler.config.num_train_timesteps,
                    (B,), device=self.device
                ).long()
                
                # Add noise to the "clean" action_label_pred_deltas
                noisy_action = self.noise_scheduler.add_noise(   
                    actions_labels=action_label_pred_deltas,
                    noise=noise,
                    timesteps=timesteps)

                # Predict the noise residual
                obs_encoding_condition = self.model("vision_encoder",obs_img=obs_image_transformed)
                
                # If goal condition, concat goal and target obs, and then infer the goal masking attention layers
                if self.goal_condition:
                    # goal_mask = (torch.rand((action_label.shape[0],)) < self.goal_mask_prob).long().to(self.device)
                    goal_mask = get_goal_mask_tensor(goal_rel_pos_to_target,self.goal_mask_prob).to(self.device)

                    linear_input = torch.concatenate([curr_rel_pos_to_target.flatten(1),
                                                normalized_prev_actions.flatten(1)], axis=1)
                    # linear_input = torch.cat((curr_rel_pos_to_target, goal_rel_pos_to_target.unsqueeze(1)), dim=1)
                    lin_encoding = self.model("linear_encoder",
                                            curr_rel_pos_to_target=linear_input)
                    
                    # lin_encoding = mask_target_context(lin_encoding, target_context_mask)
                    
                    modalities = [obs_encoding_condition, lin_encoding]
                    modal_dropout_mask = get_modal_dropout_mask(B,modalities_size=len(modalities),curr_rel_pos_to_target=curr_rel_pos_to_target,modal_dropout_prob=self.modal_dropout_prob).to(self.device)   # modify
                    
                    fused_modalities_encoding = self.model("fuse_modalities",
                                                        modalities=modalities,
                                                        mask=modal_dropout_mask)
                    
                    goal_encoding = self.model("goal_encoder",
                                            goal_rel_pos_to_target=goal_rel_pos_to_target)
                    
                    final_encoded_condition = torch.cat((fused_modalities_encoding, goal_encoding), dim=1)  # >> Concat the lin_encoding as a token too
                    final_encoded_condition = self.model("goal_masking",
                                                        final_encoded_condition=final_encoded_condition,
                                                        goal_mask = goal_mask)

                else:      # TODO: next refactoring # No Goal condition >> take the obs_encoding as the tokens
                    goal_mask = None
                    final_encoded_condition = obs_encoding_condition

                noise_pred = self.model("noise_pred",
                                        noisy_action=noisy_action,
                                        timesteps=timesteps,
                                        final_encoded_condition=final_encoded_condition)

                losses = compute_noise_losses(noise_pred=noise_pred,
                        noise=noise,
                        action_mask=action_mask)
                loss = losses["diffusion_noise_loss"]
                
            else:
                model_outputs = self.model("action_pred",
                                        obs_img=obs_image_transformed,
                                        curr_rel_pos_to_target=curr_rel_pos_to_target,
                                        goal_rel_pos_to_target=goal_rel_pos_to_target)
                action_pred = model_outputs

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
                self.scheduler.step()
                            
            # Update Exponential Moving Average of the model weights after optimizing
            if self.use_ema:
                self.ema.step(self.model.parameters())
            
            if self.model_name == "pidiff": ### TODO: add eme implementation

                if i % self.print_log_freq == 0 :
                    
                    
                    # initialize action from Gaussian noise
                    noisy_diffusion_output = torch.randn(
                        (len(final_encoded_condition), self.pred_horizon, action_dim),device=self.device)
                    diffusion_output = noisy_diffusion_output
                    
                    for k in self.noise_scheduler.timesteps():
                        # predict noise
                        noise_pred = self.model("noise_pred",
                                        noisy_action=diffusion_output,
                                        timesteps=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(self.device),
                                        final_encoded_condition=final_encoded_condition)

                        # inverse diffusion step (remove noise)
                        diffusion_output = self.noise_scheduler.remove_noise(
                            model_output=noise_pred,
                            timestep=k,
                            sample=diffusion_output
                        )

                    # action_pred = action_pred[:,:self.action_horizon,:]
                    action_pred = deltas_to_actions(deltas=diffusion_output,
                                                    pred_horizon=self.pred_horizon,
                                                    action_horizon=self.action_horizon,
                                                    learn_angle=self.learn_angle)

                    action_losses  = compute_losses(action_label=action_label,
                                                                        action_pred=action_pred,
                                                                        action_mask=action_mask,
                                                                        )
                    losses.update(action_losses)
                    
                # step lr scheduler every batch
                # this is different from standard pytorch behavior
                # TODO: check
                # if self.scheduler is not None:
                #     # scheduler calls based on the type of scheduler
                #     if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                #         self.scheduler.step(action_losses)
                #     else:


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
                action_context=normalized_prev_actions,
                action_mask=action_mask,
                goal_pos=goal_pos,
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
        self.noise_scheduler.eval()
        
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

        if self.model_name == "pidiff":
            diffusion_noise_loss = Logger(
                "diffusion_noise_loss", eval_type, window_size=self.eval_log_freq
            )
            loggers["diffusion_noise_loss"] = diffusion_noise_loss
        
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
                    obs_image_transformed,
                    curr_rel_pos_to_target,
                    goal_rel_pos_to_target,
                    action_label,
                    normalized_prev_actions,
                    goal_pos,
                    dataset_index,
                    action_mask,
                ) = data

                B = action_label.shape[0]
                action_dim = action_label.shape[-1]
                # STATE
                # visual context
                viz_images = torch.split(obs_image_transformed, 1, dim=1)
                viz_obs_image = TF.resize(viz_images[-1], VISUALIZATION_IMAGE_SIZE)
                viz_context_t0_image = TF.resize(viz_images[-self.action_context_size], VISUALIZATION_IMAGE_SIZE)

                obs_image_transformed = obs_image_transformed.to(self.device)

                # current relative target pos
                curr_rel_pos_to_target = curr_rel_pos_to_target.to(self.device)

                # GOAL
                goal_rel_pos_to_target = goal_rel_pos_to_target.to(self.device)

                # ACTION
                action_label_pred = action_label.to(self.device)
                normalized_prev_actions = normalized_prev_actions.to(self.device)


                # Take the actions horizon samples for the action loss
                action_label = action_label[:,:self.action_horizon,:]
                action_label = action_label.to(self.device)
                
                # Take the action mask. TODO: check
                action_mask = action_mask.to(self.device)

                # Infer model
                if self.model_name == "pidiff":
                    action_label_pred_deltas_pos = get_delta(actions=action_label_pred[:,:,:2]) # deltas of x,y,cos_yaw, sin_yaw
                    action_label_pred_rel_orientations = action_label_pred[:,:,2:]
                    action_label_pred_deltas = torch.cat([action_label_pred_deltas_pos, action_label_pred_rel_orientations], dim=2)
                    
                    # Take action label deltas
                    action_label_pred_deltas = get_delta(actions=action_label_pred[:,:,:2]) # deltas of x,y,cos_yaw, sin_yaw
                    
                    if self.learn_angle:
                        # We don't take the deltas of yaw
                        action_label_pred_deltas = torch.cat([action_label_pred_deltas, action_label_pred[:,:,2:]], dim=2)

                    # Sample noise to add to actions
                    noise = torch.randn(action_label_pred_deltas.shape, device=self.device)

                    # Sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, self.noise_scheduler.noise_scheduler.config.num_train_timesteps,
                        (action_label_pred_deltas.shape[0],), device=self.device
                    ).long()
                    
                    # Add noise to the "clean" action_label_pred_deltas
                    noisy_action = self.noise_scheduler.add_noise(   
                        action_label_pred_deltas, noise, timesteps)

                    # Predict the noise residual
                    obs_encoding_condition = eval_model("vision_encoder",obs_img=obs_image_transformed)

                    # If goal condition, concat goal and target obs, and then infer the goal masking attention layers
                    if self.goal_condition:

                        goal_mask = get_goal_mask_tensor(goal_rel_pos_to_target,self.goal_mask_prob).to(self.device)

                        linear_input = torch.concatenate([curr_rel_pos_to_target.flatten(1),
                                                    normalized_prev_actions.flatten(1)], axis=1)
                        # linear_input = torch.cat((curr_rel_pos_to_target, goal_rel_pos_to_target.unsqueeze(1)), dim=1)
                        lin_encoding = self.model("linear_encoder",
                                                curr_rel_pos_to_target=linear_input)
                        
                        modalities = [obs_encoding_condition, lin_encoding]
                        modal_dropout_mask = get_modal_dropout_mask(B,modalities_size=len(modalities),curr_rel_pos_to_target=curr_rel_pos_to_target,modal_dropout_prob=self.modal_dropout_prob).to(self.device)   # modify

                        fused_modalities_encoding = eval_model("fuse_modalities",
                                                            modalities=modalities,
                                                            mask=modal_dropout_mask)

                        goal_encoding = eval_model("goal_encoder",
                                                goal_rel_pos_to_target=goal_rel_pos_to_target)
                        
                        final_encoded_condition = torch.cat((fused_modalities_encoding, goal_encoding), dim=1)  # >> Concat the lin_encoding as a token too
                        # final_encoded_condition = self.model("goal_masking",
                        #                                     final_encoded_condition=final_encoded_condition,
                        #                                     goal_mask = goal_mask)
                        
                        final_encoded_condition = eval_model("goal_masking",
                                                            final_encoded_condition=final_encoded_condition,
                                                            goal_mask = goal_mask)


                    ## TODO: next refactoring
                    else:       # No Goal condition >> take the obs_encoding as the tokens
                        final_encoded_condition = obs_encoding_condition

                    noise_pred = eval_model("noise_pred",
                                        noisy_action=noisy_action,
                                        timesteps=timesteps,
                                        final_encoded_condition=final_encoded_condition)

                    losses = compute_noise_losses(noise_pred=noise_pred,
                            noise=noise,
                            action_mask=action_mask)
                    loss = losses["diffusion_noise_loss"]

                    if i % self.eval_log_freq == 0 :

                        # initialize action from Gaussian noise
                        noisy_diffusion_output = torch.randn(
                            (len(final_encoded_condition), self.pred_horizon, action_dim),device=self.device)
                        diffusion_output = noisy_diffusion_output
                        
                        for k in self.noise_scheduler.timesteps():
                            # predict noise
                            noise_pred = eval_model("noise_pred",
                                            noisy_action=diffusion_output,
                                            timesteps=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(self.device),
                                            final_encoded_condition=final_encoded_condition)

                            # inverse diffusion step (remove noise)
                            diffusion_output = self.noise_scheduler.remove_noise(
                                model_output=noise_pred,
                                timestep=k,
                                sample=diffusion_output
                            )

                        action_pred = deltas_to_actions(deltas=diffusion_output,
                                                        pred_horizon=self.pred_horizon,
                                                        action_horizon=self.action_horizon,
                                                        learn_angle=self.learn_angle)
                        
                        action_losses  = compute_losses(action_label=action_label,
                                                                                action_pred=action_pred,
                                                                                action_mask=action_mask,
                                                                            )
                        losses.update(action_losses)

                # Not in use -> TODO: refactor for another models
                else:
                    model_outputs = self.model("action_pred",
                                            obs_img=obs_image_transformed,
                                            curr_rel_pos_to_target=curr_rel_pos_to_target,
                                            goal_rel_pos_to_target=goal_rel_pos_to_target)
                    action_pred = model_outputs

                    losses = compute_losses(
                        action_label=action_label,
                        action_pred=action_pred,
                        action_mask=action_mask,
                    )
                    
                    loss = losses["total_loss"]

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
                    action_context=normalized_prev_actions,
                    action_mask=action_mask,
                    goal_pos=goal_pos,
                    dataset_index=dataset_index,
                    mode=eval_type,
                    use_latest=False,
                    wandb_increment_step=False,
                )
                
                if self.debug.max_val_steps:
                    if i==self.debug.max_val_steps:
                        break

        return action_loss_logger.average(), total_loss_logger.average()