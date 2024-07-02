from typing import List, Dict, Optional, Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
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

def get_goal_mask_tensor(goal_rel_pos_to_target,goal_mask_prob=0.5):

    goal_mask = (torch.sum(goal_rel_pos_to_target==torch.zeros_like(goal_rel_pos_to_target),axis=1) == goal_rel_pos_to_target.shape[1]).long()
    num_ones = torch.sum(goal_mask)
    total_elements = goal_mask.size(0)
    beta = (num_ones.float() / total_elements).cpu()

    probability = goal_mask_prob - beta 
    if probability > 0:
        zero_indices = (goal_mask == 0).nonzero(as_tuple=True)[0]
        random_values = torch.rand(zero_indices.size())
        mask_indices = zero_indices[random_values < probability]
        goal_mask[mask_indices] = 1
    
    return goal_mask

# def get_modal_dropout_mask(batch_size: int, modalities_size: int,curr_rel_pos_to_target:torch.tensor, modal_dropout_prob: float):
    
#     # modal_mask = (torch.sum(torch.sum(curr_rel_pos_to_target==torch.zeros_like(curr_rel_pos_to_target),axis=-1),axis=1) == curr_rel_pos_to_target.shape[1]).long()
#     # num_ones = torch.sum(modal_mask)
#     # total_elements = modal_mask.size(0)
#     # beta = (num_ones.float() / total_elements).cpu()

#     # probability = modal_dropout_prob - beta 
#     # if probability > 0:
#     #     zero_indices = (modal_mask == 0).nonzero(as_tuple=True)[0]
#     #     random_values = torch.rand(zero_indices.size())
#     #     mask_indices = zero_indices[random_values < probability]
#     #     modal_mask[mask_indices] = 1

#     # Initialize the mask tensor with ones
#     mask = torch.ones(batch_size, modalities_size, dtype=torch.int)
    
#     # Determine how many tensors to drop
#     num_to_drop = int(batch_size * modal_dropout_prob)
    
#     # Randomly select the indices of the tensors to drop
#     drop_indices = random.sample(range(batch_size), num_to_drop)
    
#     # For each selected tensor, randomly select one modality to mask
#     for idx in drop_indices:
#         modality_to_mask = random.randint(0, modalities_size - 1)
#         mask[idx, modality_to_mask] = 0
    
#     return mask

def get_modal_dropout_mask(batch_size: int, modalities_size: int, curr_rel_pos_to_target: torch.Tensor, modal_dropout_prob: float):
    # Initialize the mask tensor with ones
    mask = torch.ones(batch_size, modalities_size, dtype=torch.int)
    
    # Check for tensors that are entirely zero and set the corresponding mask value
    is_zero_tensor = torch.sum((curr_rel_pos_to_target == 0).all(axis=-1).long(),axis=-1)
    
    # Not in use!! #TODO: check
    # mask[:, 1] = 1 - is_zero_tensor  # Mask the second modality if the tensor is entirely zero
    
    # Calculate the adjusted modal dropout probability
    num_zeros = torch.sum(is_zero_tensor)
    total_elements = is_zero_tensor.size(0)
    beta = (num_zeros.float() / total_elements).cpu()

    probability = modal_dropout_prob - beta.item()
    
    if probability > 0:
        # Determine how many tensors to drop based on the adjusted probability
        num_to_drop = int(batch_size * probability)
        
        # Randomly select the indices of the tensors to drop
        zero_indices = (is_zero_tensor == 0).nonzero(as_tuple=True)[0]
        drop_indices = random.sample(zero_indices.tolist(), num_to_drop)
        
        # For each selected tensor, randomly select one modality to mask
        for idx in drop_indices:
            modality_to_mask = random.randint(0, modalities_size - 1)
            mask[idx, modality_to_mask] = 0
    
    return mask
