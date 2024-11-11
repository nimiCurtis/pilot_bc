from typing import List, Dict, Optional, Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

import os
import subprocess
import time

def assign_free_gpus(threshold_vram_usage=1500, max_gpus=2, wait=False, sleep_time=10):
    """
    Assigns free gpus to the current process via the CUDA_AVAILABLE_DEVICES env variable
    This function should be called after all imports,
    in case you are setting CUDA_AVAILABLE_DEVICES elsewhere

    Borrowed and fixed from https://gist.github.com/afspies/7e211b83ca5a8902849b05ded9a10696

    Args:
        threshold_vram_usage (int, optional): A GPU is considered free if the vram usage is below the threshold
                                            Defaults to 1500 (MiB).
        max_gpus (int, optional): Max GPUs is the maximum number of gpus to assign.
                                Defaults to 2.
        wait (bool, optional): Whether to wait until a GPU is free. Default False.
        sleep_time (int, optional): Sleep time (in seconds) to wait before checking GPUs, if wait=True. Default 10.
    """

    def _check():
        # Get the list of GPUs via nvidia-smi
        smi_query_result = subprocess.check_output(
            "nvidia-smi -q -d Memory | grep -A4 GPU", shell=True
        )
        # Extract the usage information
        gpu_info = smi_query_result.decode("utf-8").split("\n")
        gpu_info = list(filter(lambda info: "Used" in info, gpu_info))
        gpu_info = [
            int(x.split(":")[1].replace("MiB", "").strip()) for x in gpu_info
        ]  # Remove garbage
        # Keep gpus under threshold only
        free_gpus = [
            str(i) for i, mem in enumerate(gpu_info) if mem < threshold_vram_usage
        ]
        free_gpus = free_gpus[: min(max_gpus, len(free_gpus))]
        gpus_to_use = ",".join(free_gpus)
        return gpus_to_use

    while True:
        gpus_to_use = _check()
        if gpus_to_use or not wait:
            break
        print(f"No free GPUs found, retrying in {sleep_time}s")
        time.sleep(sleep_time)

    if not gpus_to_use:
        raise RuntimeError("No free GPUs found")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus_to_use

def get_gpu_memory_usage(device_id):
    """
    Returns the memory usage of the specified GPU in terms of total memory
    and allocated memory as a percentage.
    """
    total_memory = torch.cuda.get_device_properties(device_id).total_memory
    allocated_memory = torch.cuda.memory_allocated(device_id)
    return allocated_memory / total_memory

# Utils for Group Norm
def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module

def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def compute_noise_losses(
            noise_pred: torch.Tensor,
            noise: torch.Tensor,
            action_mask: torch.Tensor = None,
    ):
        """
        Compute losses for distance and action prediction.
        """

        def action_reduce(unreduced_loss: torch.Tensor):
            # Reduce over non-batch dimensions to get loss per batch element
            while unreduced_loss.dim() > 1:
                unreduced_loss = unreduced_loss.mean(dim=-1)
                # unreduced_loss = unreduced_loss.sum(dim=-1)

            assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
            return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

        # L2 loss
        diffusion_noise_loss = action_reduce(F.mse_loss(noise_pred, noise, reduction="none"))
            
        # Total loss
        # loss = alpha * dist_loss + (1-alpha) * diffusion_loss

        results = {
            "diffusion_noise_loss": diffusion_noise_loss,
        }

        ## For now
        # total_loss = diffusion_noise_loss 
        # results["total_loss"] = total_loss

        return results

def compute_losses(
            action_label: torch.Tensor,
            action_pred: torch.Tensor,
            action_mask: torch.Tensor = None,
            control_magnitude: torch.Tensor = None,
    ):
        """
        Compute losses for distance and action prediction.
        """

        def action_reduce(unreduced_loss: torch.Tensor):
            # Reduce over non-batch dimensions to get loss per batch element
            while unreduced_loss.dim() > 1:
                # unreduced_loss = unreduced_loss.sum(dim=-1)
                unreduced_loss = unreduced_loss.mean(dim=-1)

            assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
            return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

        # Mask out invalid inputs (for negatives, or when the distance between obs and goal is large)
        # This is the actual losses
        assert action_pred.shape == action_label.shape, f"{action_pred.shape} != {action_label.shape}"
        action_loss = action_reduce(F.mse_loss(action_pred, action_label, reduction="none"))

        # Other losses for logger
        action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
            action_pred[:, :, :2], action_label[:, :, :2], dim=-1
        ))
        multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
            torch.flatten(action_pred[:, :, :2], start_dim=1),
            torch.flatten(action_label[:, :, :2], start_dim=1),
            dim=-1,
        ))

        results = {
            "action_loss": action_loss,
            "action_waypts_cos_sim": action_waypts_cos_similairity,
            "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim,
        }

        learn_angle = action_pred.shape[-1] == 4
        if learn_angle:
            action_orien_cos_sim = action_reduce(F.cosine_similarity(
                action_pred[:, :, 2:], action_label[:, :, 2:], dim=-1
            ))
            multi_action_orien_cos_sim = action_reduce(F.cosine_similarity(
                torch.flatten(action_pred[:, :, 2:], start_dim=1),
                torch.flatten(action_label[:, :, 2:], start_dim=1),
                dim=-1,
            )
            )
            results["action_orien_cos_sim"] = action_orien_cos_sim
            results["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim

        total_loss = action_loss if control_magnitude is None else control_magnitude*action_loss
        results["total_loss"] = total_loss

        return results



def load_model(model,model_name, checkpoint_path:str, model_version:str = "best_model"):
        """
        Load a pre-trained model.

        Args:
            model_name (str): The name of the pre-trained model.
        """
        model_path = os.path.join(checkpoint_path, model_name, f"{model_version}.pth")
        checkpoint = torch.load(model_path,map_location='cpu')

        state_dict = checkpoint["model_state_dict"]
        model.load_state_dict(state_dict, strict=False)


