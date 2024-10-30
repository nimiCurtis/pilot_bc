import numpy as np

from omegaconf import DictConfig
from typing import List
import torch
from torch import nn
from torch.utils.data import DataLoader


from pilot_train.training.trainers.pidiff_trainer import PiDiffTrainer
from pilot_train.training.trainers.vint_trainer import ViNTTrainer
from pilot_train.training.trainers.cnn_mlp_trainer import CNNMLPTrainer


# Registry of available trainers
TRAINERS = {
    'pidiff': PiDiffTrainer,
    'vint': ViNTTrainer,
    'cnn_mlp': CNNMLPTrainer
}

def register_trainer(model: nn.Module,
                optimizer: torch.optim,
                scheduler,
                dataloader: DataLoader,
                test_dataloaders: List[DataLoader],
                device,
                training_cfg: DictConfig,
                data_cfg: DictConfig, 
                log_cfg: DictConfig,
                datasets_cfg: DictConfig):
    
    model_name = model.module.name if hasattr(model, "module") else model.name
    
    trainer_class = TRAINERS[model_name]
    
    trainer = trainer_class(model,
                optimizer,
                scheduler,
                dataloader,
                test_dataloaders,
                device,
                training_cfg,
                data_cfg, 
                log_cfg,
                datasets_cfg
            )   
    
    return trainer