import torch
import numpy as np
import time
import random

def tic():
    return time.time()

def toc(t):
    return float(tic()) - float(t)

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()

def from_numpy(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).float()

# normalize data
def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    # back to [0, 1]
    ndata = (ndata + 1) / 2
    # back to [min, max] 
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

#Ours
def get_delta(actions):
    if isinstance(actions, np.ndarray):
        # Append zeros to the first action for NumPy array
        ex_actions = np.concatenate([np.zeros((1, actions.shape[-1])), actions], axis=0)
        delta = ex_actions[1:, :] - ex_actions[:-1, :]
    elif isinstance(actions, torch.Tensor):
        # Append zeros to the first action for PyTorch tensor
        ex_actions = torch.cat([torch.zeros((actions.shape[0],1, actions.shape[-1]), dtype=actions.dtype, device=actions.device), actions], dim=1)
        delta = ex_actions[:,1:, :] - ex_actions[:,:-1, :]
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor")

    
    return delta


def calculate_sin_cos(waypoints):
    """
    Calculate sin and cos of the angle for waypoints.

    Args:
        waypoints: A NumPy array or PyTorch tensor of waypoints. Expected shape is [N, 3] where
                   the last dimension contains [x, y, angle].

    Returns:
        A NumPy array or PyTorch tensor (matching the input type) of waypoints with sin and cos
        of the angle appended. Output shape is [N, 4], containing [x, y, sin(angle), cos(angle)].
    """
    library = torch if isinstance(waypoints, torch.Tensor) else np

    assert waypoints.shape[1] == 3, "Waypoints should have shape [N, 3]."

    angle = waypoints[:, 2]
    sin_angle = library.sin(angle)
    cos_angle = library.cos(angle)

    if library is torch:
        angle_repr = torch.stack((cos_angle, sin_angle), dim=1)
        return torch.cat((waypoints[:, :2], angle_repr), dim=1)
    else:
        angle_repr = np.stack((cos_angle, sin_angle), axis=1)
        return np.concatenate((waypoints[:, :2], angle_repr), axis=1)

def xy_to_d_cos_sin(xy):
    
    if len(xy.shape) == 1:
        d = np.linalg.norm(xy)
        angle = np.arctan2(xy[1], xy[0])
    else:
        d = np.linalg.norm(xy, axis=1)
        angle = np.arctan2(xy[:,1], xy[:,0])
    
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    d_cos_sin = np.stack((d.T,cos_angle.T,sin_angle.T)).T
    
    return d_cos_sin

def get_goal_mask_tensor(goal_rel_pos_to_target,goal_mask_prob=0.0):

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