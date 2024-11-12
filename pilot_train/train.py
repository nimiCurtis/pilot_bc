import os
import wandb
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from pilot_train.training.trainers.basic_trainer import BasicTrainer as bt
from pilot_train.training.trainer import build_trainer
from pilot_config.config import get_main_config_dir, split_main_config
from pilot_utils.utils import tic, toc
from pilot_utils.transforms import ObservationTransform
from pilot_utils.train.train_utils import get_gpu_memory_usage, load_model
from pilot_deploy.inference import CKPTH_PATH, get_inference_config


def train(cfg:DictConfig):
    
    # Get configs
    training_cfg, device_cfg,  data_cfg, datasets_cfg, policy_model_cfg, vision_encoder_model_cfg, linear_encoder_model_cfg, log_cfg =  split_main_config(cfg)

    if training_cfg.fine_tune.enable:
        data_cfg, _, policy_model_cfg, vision_encoder_model_cfg, linear_encoder_model_cfg, _ = get_inference_config(model_name=training_cfg.fine_tune.model)

    # Device management
    if torch.cuda.is_available() and device_cfg == 'cuda':
        available_gpus = list(range(torch.cuda.device_count()))
        print("Available GPU IDs:", available_gpus)
        
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        
        if "gpu_ids" not in training_cfg:
            OmegaConf.update(training_cfg, "gpu_ids", [0], force_add=True)
        elif isinstance(training_cfg.gpu_ids, int):
            OmegaConf.update(training_cfg, "gpu_ids", [training_cfg.gpu_ids])

        # Check if gpu_ids are valid and meet the memory requirement
        valid_gpu_ids = []
        for gpu_id in training_cfg.gpu_ids:
            if gpu_id in available_gpus:
                memory_usage = get_gpu_memory_usage(gpu_id)
                if memory_usage <= 0.25:
                    valid_gpu_ids.append(gpu_id)
        
        if not valid_gpu_ids:
            print("No valid gpu. Exit!")
            exit()
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in valid_gpu_ids])
            print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
            
            first_gpu_id = valid_gpu_ids[0] if valid_gpu_ids else 0
            device = torch.device(f"cuda:{first_gpu_id}")
    else:
        print("Using cpu")
        device = torch.device("cpu")

    if "seed" in training_cfg: # Reproducing performance
        np.random.seed(training_cfg.seed)
        torch.manual_seed(training_cfg.seed)
        cudnn.deterministic = True
    cudnn.benchmark = True  # good if input sizes don't vary

    # Transform
    transform = ObservationTransform(data_cfg=data_cfg)

    # Training Getters
    train_dataloader, test_dataloaders = bt.get_dataloaders(datasets_cfg=datasets_cfg,
                                                                data_cfg=data_cfg,
                                                                training_cfg=training_cfg,
                                                                transform=transform)
    model = bt.get_model(
        policy_model_cfg = policy_model_cfg,
        vision_encoder_model_cfg = vision_encoder_model_cfg,
        linear_encoder_model_cfg = linear_encoder_model_cfg,
        data_cfg = data_cfg
        )
    model_name = model.name
    print(f"Model Type: {model_name}")
    model.count_parameters()

    if training_cfg.fine_tune.enable:
        load_model(model,
                model_name=training_cfg.fine_tune.model,
                checkpoint_path=CKPTH_PATH)

    # Multi-GPU
    model = model.to(device)
    if device.type == 'cuda' and len(training_cfg.gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=training_cfg.gpu_ids)
        model = model.to(device)

    ##  Set Pilot Trainer  ## 
    pilot_trainer = build_trainer(model=model,
        dataloader=train_dataloader,
        test_dataloaders=test_dataloaders,
        device=device,
        training_cfg=training_cfg,
        data_cfg=data_cfg,
        log_cfg=log_cfg,
        datasets_cfg=datasets_cfg
    )
    
    print()
    print("START TRAINING")
    start_time = tic()
    pilot_trainer.run()
    
    print("FINISHED TRAINING")
    print(f"TRAINING TIME: {toc(start_time)/60} [minutes]")

if __name__ == "__main__":

    @hydra.main(version_base=None, config_path=get_main_config_dir(), config_name="train_pilot_policy")
    def main(cfg: DictConfig):
        torch.multiprocessing.set_start_method("spawn")

        wandb_cfg = cfg.log.wandb

        if wandb_cfg.run.enable:
            wandb.login()
            wandb.init(
                project=wandb_cfg.setup.project,
                settings=wandb.Settings(start_method="fork"),
                entity=wandb_cfg.setup.entity 
            )
            wandb.run.name = wandb_cfg.run.name
            
            if wandb.run:
                wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

        print("******** Training Config: *********")
        print(OmegaConf.to_yaml(cfg, resolve=True))
        print("********************\n")
        
        # Train
        train(cfg)

    main()
