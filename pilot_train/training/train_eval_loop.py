import wandb
import os
import numpy as np
from typing import List, Optional, Dict
from prettytable import PrettyTable
from omegaconf import DictConfig, OmegaConf

from pilot_train.training.train_utils import train, evaluate
#from pilot_train.training.train_utils import train_nomad, evaluate_nomad

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms


def train_eval_loop(
            model: nn.Module,
            optimizer: torch.optim,
            scheduler,
            dataloader: DataLoader,
            test_dataloaders: List[DataLoader],
            transform: transforms,
            device,
            training_cfg: DictConfig,
            data_cfg: DictConfig,
            log_cfg: DictConfig):

    """
    Train and evaluate the model for several epochs (vint or gnm models)

    Args:
        model: model to train
        optimizer: optimizer to use
        scheduler: learning rate scheduler to use
        dataloader: dataloader for train dataset
        test_dataloaders: dict of dataloaders for testing
        transform: transform to apply to images
        epochs: number of epochs to train
        device: device to train on
        project_folder: folder to save checkpoints and logs
        normalized: whether to normalize the action space or not
        wandb_log_freq: frequency of logging to wandb
        print_log_freq: frequency of printing to console
        image_log_freq: frequency of logging images to wandb
        num_images_log: number of images to log to wandb
        current_epoch: epoch to start training from
        alpha: tradeoff between distance and action loss
        learn_angle: whether to learn the angle or not
        use_wandb: whether to log to wandb or not
        eval_fraction: fraction of training data to use for evaluation
    """
    
    # Log config
    project_folder=log_cfg.project_folder
    print_log_freq=log_cfg.print_log_freq
    wandb_log_freq = log_cfg.wandb.run.log_freq
    image_log_freq=log_cfg.image_log_freq
    num_images_log=log_cfg.num_images_log
    use_wandb=log_cfg.wandb.run.enable
    eval_fraction=log_cfg.eval_fraction
    
    # Training config
    alpha=training_cfg.alpha if training_cfg.goal_condition else 0
    beta=training_cfg.beta
    goal_condition = training_cfg.goal_condition
    current_epoch=training_cfg.current_epoch
    epochs=training_cfg.epochs
    
    # Data config
    normalized=data_cfg.normalize
    learn_angle=data_cfg.learn_angle
    
    assert 0 <= alpha <= 1
    latest_path = os.path.join(project_folder, f"latest.pth") #change

    for epoch in range(current_epoch, current_epoch + epochs):
        print(
            f"Start ViNT Training Epoch {epoch}/{current_epoch + epochs - 1}"
        )
        train(
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            transform=transform,
            device=device,
            project_folder=project_folder,
            normalized=normalized,
            epoch=epoch,
            alpha=alpha,
            beta=beta,
            learn_angle=learn_angle,
            print_log_freq=print_log_freq,
            wandb_log_freq=wandb_log_freq,
            image_log_freq=image_log_freq,
            num_images_log=num_images_log,
            use_wandb=use_wandb,
            goal_condition=goal_condition
        )

        avg_total_test_loss = []
        for dataset_type in test_dataloaders:
            print(
                f"Start {dataset_type} ViNT Testing Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            loader = test_dataloaders[dataset_type]

            test_dist_loss, test_action_loss, total_eval_loss = evaluate(
                eval_type=dataset_type,
                model=model,
                dataloader=loader,
                transform=transform,
                device=device,
                project_folder=project_folder,
                normalized=normalized,
                epoch=epoch,
                alpha=alpha,
                beta=beta,
                learn_angle=learn_angle,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                eval_fraction=eval_fraction,
                goal_condition=goal_condition
            )

            avg_total_test_loss.append(total_eval_loss)

        checkpoint = {
            "epoch": epoch,
            "model": model,
            "optimizer": optimizer,
            "avg_total_test_loss": np.mean(avg_total_test_loss),
            "scheduler": scheduler
        }
        # log average eval loss
        wandb.log({}, commit=False)

        if scheduler is not None:
            # scheduler calls based on the type of scheduler
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(np.mean(avg_total_test_loss))
            else:
                scheduler.step()
        wandb.log({
            "avg_total_test_loss": np.mean(avg_total_test_loss),
            "lr": optimizer.param_groups[0]["lr"],
        }, commit=False)

        numbered_path = os.path.join(project_folder, f"{epoch}.pth")
        torch.save(checkpoint, latest_path)
        torch.save(checkpoint, numbered_path)  # keep track of model at every epoch

    # Flush the last set of eval logs
    wandb.log({})
    print()


def load_model(model, model_type, checkpoint: dict) -> None:
    """Load model from checkpoint."""
    if model_type == "nomad":
        state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        loaded_model = checkpoint["model"]
        try:
            state_dict = loaded_model.module.state_dict()
            model.load_state_dict(state_dict, strict=False)
        except AttributeError as e:
            state_dict = loaded_model.state_dict()
            model.load_state_dict(state_dict, strict=False)


def load_ema_model(ema_model, state_dict: dict) -> None:
    """Load model from checkpoint."""
    ema_model.load_state_dict(state_dict)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    # print(table)
    print(f"Total Trainable Params: {total_params / 1e6:.2f}M")
    return total_params