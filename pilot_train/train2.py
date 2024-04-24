import os
import wandb
import argparse
import numpy as np
import yaml
import time
import pdb
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam, AdamW, SGD
from torchvision import transforms
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler

# from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
# from diffusers.optimization import get_scheduler

from pilot_config.config import get_config_dir
"""
IMPORT YOUR MODEL HERE
"""

# from vint_train.models.gnm.gnm import GNM
# from vint_train.models.vint.vint import ViNT
# from vint_train.models.vint.vit import ViT
# from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
# from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
# from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

from pilot_train.models.policy.model_registry import get_policy_model

# from vint_train.data.vint_dataset import ViNT_Dataset
from pilot_train.training.train_eval_loop import (
    train_eval_loop,
    load_model,
)

from pilot_train.data.pilot_dataset import PilotDataset

def load_config(cfg:DictConfig)->Tuple[DictConfig]:
    missings = OmegaConf.missing_keys(cfg)
    # Assertion to check if the set is empty
    assert not missings, f"Missing configs: {missings}, please check the main config!"

    configs = {}
    for key in cfg.keys():
        assert key in ['training', 'data', 'log', 'encoder_model', 'policy_model', 'datasets']\
        ,f"{key} is missing in config please check tha main config!"

    return cfg.training, cfg.data, cfg.datasets, cfg.policy_model, cfg.encoder_model, cfg.log

def get_model(policy_model_cfg, encoder_model_cfg , training_cfg, data_cfg ):
    
    return get_policy_model(
            policy_model_cfg = policy_model_cfg,
            encoder_model_cfg = encoder_model_cfg,
            training_cfg = training_cfg,
            data_cfg = data_cfg
            )

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

def train(cfg:DictConfig):
    
    training_cfg, data_cfg, datasets_cfg, policy_model_cfg, encoder_model_cfg, log_cfg =  load_config(cfg)

    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in training_cfg:
            OmegaConf.update(training_cfg, "gpu_ids",[0], force_add = True)
        elif type(training_cfg.gpu_ids) == int:
            OmegaConf.update(training_cfg, "gpu_ids",[training_cfg.gpu_ids])
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in training_cfg.gpu_ids]
        )
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")

    first_gpu_id = training_cfg.gpu_ids[0]
    device = torch.device(
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    if "seed" in training_cfg:
        np.random.seed(training_cfg.seed)
        torch.manual_seed(training_cfg.seed)
        cudnn.deterministic = True

    cudnn.benchmark = True  # good if input sizes don't vary
    transform = ([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform = transforms.Compose(transform)

    # Load the data
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

    # Create the model
    # if config["model_type"] == "gnm":
    #     model = GNM(
    #         config["context_size"],
    #         config["len_traj_pred"],
    #         config["learn_angle"],
    #         config["obs_encoding_size"],
    #         config["goal_encoding_size"],
    #     )
    if policy_model_cfg.name == "vint":
        model = get_model(
            policy_model_cfg = policy_model_cfg,
            encoder_model_cfg = encoder_model_cfg,
            training_cfg = training_cfg,
            data_cfg = data_cfg
            )

    # elif config["model_type"] == "nomad":
    #     if config["vision_encoder"] == "nomad_vint":
    #         vision_encoder = NoMaD_ViNT(
    #             obs_encoding_size=config["encoding_size"],
    #             context_size=config["context_size"],
    #             mha_num_attention_heads=config["mha_num_attention_heads"],
    #             mha_num_attention_layers=config["mha_num_attention_layers"],
    #             mha_ff_dim_factor=config["mha_ff_dim_factor"],
    #         )
    #         vision_encoder = replace_bn_with_gn(vision_encoder)
    #     elif config["vision_encoder"] == "vib": 
    #         vision_encoder = ViB(
    #             obs_encoding_size=config["encoding_size"],
    #             context_size=config["context_size"],
    #             mha_num_attention_heads=config["mha_num_attention_heads"],
    #             mha_num_attention_layers=config["mha_num_attention_layers"],
    #             mha_ff_dim_factor=config["mha_ff_dim_factor"],
    #         )
    #         vision_encoder = replace_bn_with_gn(vision_encoder)
    #     elif config["vision_encoder"] == "vit": 
    #         vision_encoder = ViT(
    #             obs_encoding_size=config["encoding_size"],
    #             context_size=config["context_size"],
    #             image_size=config["image_size"],
    #             patch_size=config["patch_size"],
    #             mha_num_attention_heads=config["mha_num_attention_heads"],
    #             mha_num_attention_layers=config["mha_num_attention_layers"],
    #         )
    #         vision_encoder = replace_bn_with_gn(vision_encoder)
    #     else: 
    #         raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")
            
    #     noise_pred_net = ConditionalUnet1D(
    #             input_dim=2,
    #             global_cond_dim=config["encoding_size"],
    #             down_dims=config["down_dims"],
    #             cond_predict_scale=config["cond_predict_scale"],
    #         )
    #     dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
        
    #     model = NoMaD(
    #         vision_encoder=vision_encoder,
    #         noise_pred_net=noise_pred_net,
    #         dist_pred_net=dist_pred_network,
    #     )

    #     noise_scheduler = DDPMScheduler(
    #         num_train_timesteps=config["num_diffusion_iters"],
    #         beta_schedule='squaredcos_cap_v2',
    #         clip_sample=True,
    #         prediction_type='epsilon'
    #     )
    # else:
    #     raise ValueError(f"Model {config['model']} not supported")

    # if config["clipping"]:
    #     print("Clipping gradients to", config["max_norm"])
    #     for p in model.parameters():
    #         if not p.requires_grad:
    #             continue
    #         p.register_hook(
    #             lambda grad: torch.clamp(
    #                 grad, -1 * config["max_norm"], config["max_norm"]
    #             )
    #         )

    lr = float(training_cfg.lr)
    optimizer_name = training_cfg.optimizer
    optimizer = get_optimizer(optimizer_name, model=model, lr=lr)
    scheduler = get_scheduler(training_cfg = training_cfg, optimizer=optimizer, lr=lr) if "scheduler" in training_cfg else None

    current_epoch = 0
    # if "load_run" in config:
    #     load_project_folder = os.path.join("logs", config["load_run"])
    #     print("Loading model from ", load_project_folder)
    #     latest_path = os.path.join(load_project_folder, "latest.pth")
    #     latest_checkpoint = torch.load(latest_path) #f"cuda:{}" if torch.cuda.is_available() else "cpu")
    #     load_model(model, config["model_type"], latest_checkpoint)
    #     if "epoch" in latest_checkpoint:
    #         current_epoch = latest_checkpoint["epoch"] + 1

    # Multi-GPU
    if len(training_cfg.gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=training_cfg.gpu_ids)
    model = model.to(device)

    # if "load_run" in config:  # load optimizer and scheduler after data parallel
    #     if "optimizer" in latest_checkpoint:
    #         optimizer.load_state_dict(latest_checkpoint["optimizer"].state_dict())
    #     if scheduler is not None and "scheduler" in latest_checkpoint:
    #         scheduler.load_state_dict(latest_checkpoint["scheduler"].state_dict())

    ### continue form here!!!! 
    
    if policy_model_cfg.name == "vint":
        train_eval_loop(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            device=device,
            training_cfg=training_cfg,
            data_cfg=data_cfg,
            log_cfg=log_cfg)

    # else:
    #     train_eval_loop_nomad(
    #         train_model=config["train"],
    #         model=model,
    #         optimizer=optimizer,
    #         lr_scheduler=scheduler,
    #         noise_scheduler=noise_scheduler,
    #         train_loader=train_loader,
    #         test_dataloaders=test_dataloaders,
    #         transform=transform,
    #         goal_mask_prob=config["goal_mask_prob"],
    #         epochs=config["epochs"],
    #         device=device,
    #         project_folder=config["project_folder"],
    #         print_log_freq=config["print_log_freq"],
    #         wandb_log_freq=config["wandb_log_freq"],
    #         image_log_freq=config["image_log_freq"],
    #         num_images_log=config["num_images_log"],
    #         current_epoch=current_epoch,
    #         alpha=float(config["alpha"]),
    #         use_wandb=config["use_wandb"],
    #         eval_fraction=config["eval_fraction"],
    #         eval_freq=config["eval_freq"],
    #     )

    print("FINISHED TRAINING")

@hydra.main( version_base=None ,
        config_path= get_config_dir(),
        config_name = "train_pilot_policy")
def main(cfg:DictConfig):
    torch.multiprocessing.set_start_method("spawn")

    log_cfg = cfg.log
    wandb_cfg =  log_cfg.wandb

    if wandb_cfg.run.enable:
        wandb.login()
        wandb.init(
            project=wandb_cfg.setup.project,
            settings=wandb.Settings(start_method="fork"),
            entity=wandb_cfg.setup.entity, # TODO: change this to your wandb entity
        )
        wandb.run.name = wandb_cfg.run.name
        # update the wandb args with the training configurations
        if wandb.run:
            wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

    print("******** Training Config: *********")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("********************\n")
    train(cfg)

if __name__ == "__main__":
    main()
