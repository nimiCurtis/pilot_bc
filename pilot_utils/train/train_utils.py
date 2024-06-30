from typing import List, Dict, Optional, Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    beta = num_ones.float() / total_elements

    probability = goal_mask_prob - beta 
    if probability > 0:
        zero_indices = (goal_mask == 0).nonzero(as_tuple=True)[0]
        random_values = torch.rand(zero_indices.size(), device='cuda:0')
        mask_indices = zero_indices[random_values < probability]
        goal_mask[mask_indices] = 1
    
    return goal_mask