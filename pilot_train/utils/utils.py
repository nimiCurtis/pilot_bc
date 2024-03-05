
import torch
import numpy as np

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
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

#VinT
# def get_delta(actions):
#     # append zeros to first action
#     ex_actions = np.concatenate([np.zeros((actions.shape[0], 1, actions.shape[-1])), actions], axis=1)
#     delta = ex_actions[:, 1:] - ex_actions[:, :-1]
#     return delta

#Ours
def get_delta(actions):
    # append zeros to first action
    ex_actions = np.concatenate([np.zeros((1, actions.shape[-1])), actions], axis=0)
    delta = ex_actions[1:,:] - ex_actions[:-1,:]
    return delta

def get_action(normalized_action_deltas, action_stats):

    ndeltas = normalized_action_deltas
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
    ndeltas = to_numpy(ndeltas)
    ndeltas = unnormalize_data(ndeltas, action_stats)
    actions = np.cumsum(ndeltas, axis=1)
    return from_numpy(actions)


