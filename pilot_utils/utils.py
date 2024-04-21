import torch
import numpy as np
import time


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
    # append zeros to first action
    ex_actions = np.concatenate([np.zeros((1, actions.shape[-1])), actions], axis=0)
    delta = ex_actions[1:,:] - ex_actions[:-1,:]
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

