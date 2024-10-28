import torch
import numpy as np
import time
import random
import torch.nn.functional as F

def tic():
    """
    Returns the current time.
    """
    return time.time()

def toc(t):
    """
    Returns the elapsed time since the input time t.
    
    Args:
        t (float): The starting time.
    
    Returns:
        float: The elapsed time.
    """
    return float(tic()) - float(t)

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a PyTorch tensor to a NumPy array.

    Args:
        tensor (torch.Tensor): The tensor to convert.

    Returns:
        np.ndarray: The converted NumPy array.
    """
    return tensor.detach().cpu().numpy()

def from_numpy(array: np.ndarray) -> torch.Tensor:
    """
    Converts a NumPy array to a PyTorch tensor.

    Args:
        array (np.ndarray): The array to convert.

    Returns:
        torch.Tensor: The converted tensor.
    """
    return torch.from_numpy(array).float()

def is_tensor(x):
    """
    Checks if the input is a PyTorch tensor.

    Args:
        x: The input to check.

    Returns:
        bool: True if the input is a PyTorch tensor, False otherwise.
    """
    return torch.is_tensor(x)


def normalize_data(data, stats, norm_type="maxmin"):
    """
    Normalizes the data to the range [-1, 1].

    Args:
        data (np.ndarray): The data to normalize.
        stats (dict): The statistics for normalization.

    Returns:
        np.ndarray: The normalized data.
    """
    
    if norm_type == "maxmin":
        # Normalize to [0,1]
        ndata = (data - stats['min']) / (stats['max'] - stats['min'])
        # Normalize to [-1, 1]
        ndata = ndata * 2 - 1
    elif norm_type == "standard":
        # Standardize using mean and standard deviation
        ndata = (data - stats['mean']) / stats['std']
    return ndata


def unnormalize_data(ndata, stats, norm_type="standard"):
    """
    Unstandardizes the data back to its original scale.

    Args:
        standardized_data (np.ndarray): The standardized data to unstandardize.
        stats (dict): The statistics used for standardization, should contain 'mean' and 'std'.

    Returns:
        np.ndarray: The unstandardized data.
    """
    if norm_type == "maxmin":
        # Back to [0, 1]
        ndata = (ndata + 1) / 2
        # Back to [min, max]
        data = ndata * (stats['max'] - stats['min']) + stats['min']
        
    elif norm_type == "standard":
        # Unstandardize using mean and standard deviation
        data = ndata * stats['std'] + stats['mean']
    return data

def get_delta(actions):
    """
    Computes the delta (difference) between consecutive actions.

    Args:
        actions (np.ndarray or torch.Tensor): The actions to compute deltas for.

    Returns:
        np.ndarray or torch.Tensor: The computed deltas.
    """
    if isinstance(actions, np.ndarray):
        # Append zeros to the first action for NumPy array
        ex_actions = np.concatenate([np.zeros((1, actions.shape[-1])), actions], axis=0)
        delta = ex_actions[1:, :] - ex_actions[:-1, :]
    elif isinstance(actions, torch.Tensor):
        if len(actions.shape)>2:
            # Append zeros to the first action for PyTorch tensor
            ex_actions = torch.cat([torch.zeros((actions.shape[0],1, actions.shape[-1]), dtype=actions.dtype, device=actions.device), actions], dim=1)
            delta = ex_actions[:,1:, :] - ex_actions[:,:-1, :]
        else:
            ex_actions = torch.cat([torch.zeros((1, actions.shape[-1]), dtype=actions.dtype, device=actions.device), actions], dim=0)
            delta = ex_actions[1:, :] - ex_actions[:-1, :]
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor")

    
    return delta


def calculate_sin_cos(waypoints):
    """
    Calculate sin and cos of the angle for waypoints.

    Args:
        waypoints: A NumPy array or PyTorch tensor of waypoints. Expected shape is [N, 3] where
                the last dimension contains [x, y, angle], or [3] for 1D input.

    Returns:
        A NumPy array or PyTorch tensor (matching the input type) of waypoints with sin and cos
        of the angle appended. Output shape is [N, 4] for 2D input, containing [x, y, sin(angle), cos(angle)],
        or [4] for 1D input.
    """
    library = torch if isinstance(waypoints, torch.Tensor) else np

    if waypoints.ndim == 1:
        assert waypoints.shape[0] == 3, "1D Waypoints should have shape [3]."
        angle = waypoints[2]
        sin_angle = library.sin(angle)
        cos_angle = library.cos(angle)
        return library.concatenate((waypoints[:2], library.array([cos_angle, sin_angle])))
    
    elif waypoints.ndim == 2:
        assert waypoints.shape[1] == 3, "2D Waypoints should have shape [N, 3]."
        angle = waypoints[:, 2]
        sin_angle = library.sin(angle)
        cos_angle = library.cos(angle)

        if library is torch:
            angle_repr = torch.stack((cos_angle, sin_angle), dim=1)
            return torch.cat((waypoints[:, :2], angle_repr), dim=1)
        else:
            angle_repr = np.stack((cos_angle, sin_angle), axis=1)
            return np.concatenate((waypoints[:, :2], angle_repr), axis=1)
    elif waypoints.ndim == 3:
        angle = waypoints[:, :, 2]
        sin_angle = library.sin(angle)
        cos_angle = library.cos(angle)

        if library is torch:
                    angle_repr = torch.stack((cos_angle, sin_angle), dim=2)
                    return torch.cat((waypoints[:,:, :2], angle_repr), dim=2)
        else:
                    angle_repr = np.stack((cos_angle, sin_angle), axis=2)
                    return np.concatenate((waypoints[:, :, :2], angle_repr), axis=2)

def xy_to_d_cos_sin(xy):
    """
    Converts XY coordinates to distance, cosine, and sine of the angle.

    Args:
        xy (np.ndarray): The XY coordinates to convert.

    Returns:
        np.ndarray: The distance, cosine, and sine of the angle.
    """
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
    """
    Generates a goal mask tensor with a specified probability.

    Args:
        goal_rel_pos_to_target (torch.Tensor): Relative positions to target.
        goal_mask_prob (float): Probability of masking the goal.

    Returns:
        torch.Tensor: The generated goal mask tensor.
    """
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

def actions_forward_pass(actions,action_stats, learn_angle, norm_type = "standard"):
    """
    Forward pass for actions, normalizing and converting deltas to trajectory.

    Args:
        actions: A NumPy array or PyTorch tensor of actions. Expected shape is [batch, sequence_len, features_len].
        action_stats: A dictionary containing statistics for normalization.
        learn_angle: Whether to learn the angle.

    Returns:
        A NumPy array or PyTorch tensor (matching the input type) of normalized actions.
    """
    
    library = torch if isinstance(actions, torch.Tensor) else np

    # Initialize normalized_actions with the original actions
    normalized_actions = actions

    # Normalize and compute deltas if actions contain multiple dimensions
    if len(actions.shape) > 1:
        # Compute deltas for the first two dimensions
        actions_deltas = get_delta(actions[:, :2])
        
        # Normalize deltas based on provided statistics
        normalized_actions_deltas = normalize_data(actions_deltas, action_stats['pos'], norm_type=norm_type)
        
        # Compute cumulative sum for normalized trajectory
        normalized_actions[:, :2] = library.cumsum(normalized_actions_deltas, axis=0)
    else:
        # Normalize actions for one-dimensional case
        normalized_actions[:2] = normalize_data(actions[:2], action_stats['pos'], norm_type=norm_type)

    if learn_angle:
        # Calculate sine and cosine for the angle
        normalized_actions = calculate_sin_cos(normalized_actions)

    return normalized_actions


def get_modal_dropout_mask(batch_size: int, modalities_size: int, curr_rel_pos_to_target: torch.Tensor, modal_dropout_prob: float):
    """
    Generates a dropout mask for modalities.

    Args:
        batch_size (int): The size of the batch.
        modalities_size (int): The number of modalities.
        curr_rel_pos_to_target (torch.Tensor): Current relative positions to target.
        modal_dropout_prob (float): Probability of dropping a modality.

    Returns:
        torch.Tensor: The generated dropout mask.
    """
    # Initialize the mask tensor with ones
    mask = torch.ones(batch_size, modalities_size, dtype=torch.int)
    
    # Check for tensors that are entirely zero and set the corresponding mask value
    is_zero_tensor = torch.sum((curr_rel_pos_to_target == 0).all(axis=-1).long(), axis=-1)
    
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

def get_action_stats(properties, waypoint_spacing):
        """
        Retrieves action statistics based on robot properties and waypoint spacing.

        Args:
            properties (dict): Robot properties.
            waypoint_spacing (int): Spacing between waypoints.

        Returns:
            dict: Action statistics.
        """
        
        frame_rate = properties['frame_rate']
        
        # List of properties to convert if they're lists
        velocity_properties = ['max_lin_vel', 'min_lin_vel', 'mean_lin_vel', 'std_lin_vel']

        # Convert properties in 'properties' dict to numpy arrays if they are lists
        for prop in velocity_properties:
            if isinstance(properties[prop], list):
                properties[prop] = np.array(properties[prop])
        
        max_lin_vel = properties['max_lin_vel']
        min_lin_vel = properties['min_lin_vel']
        mean_lin_vel = properties['mean_lin_vel']
        std_lin_vel = properties['std_lin_vel']

        ang_vel_lim = properties['max_ang_vel']
        
        return {'pos': {
                        'max': (max_lin_vel / frame_rate)*waypoint_spacing,
                        'min': (min_lin_vel /frame_rate)*waypoint_spacing,
                        'mean': (mean_lin_vel / frame_rate)*waypoint_spacing,
                        'std': (std_lin_vel /frame_rate)*waypoint_spacing,
                        },
                'yaw': {'max': (ang_vel_lim /frame_rate)*waypoint_spacing,
                        'min': -(ang_vel_lim /frame_rate)*waypoint_spacing }}


# def clip_angle(theta: float) -> float:
#     """
#     Clips an angle to the range [-π, π].

#     Args:
#         theta (float): Input angle in radians.

#     Returns:
#         float: Clipped angle within the range [-π, π].
#     """
#     theta %= 2 * np.pi
#     if -np.pi < theta < np.pi:
#         return theta
#     return theta - 2 * np.pi

# def clip_angles(angles):
#     # Use modulo operation to keep the angles in the range [-pi, pi]
#     return (angles + np.pi) % (2 * np.pi) - np.pi

def clip_angles(angles):
    return np.arctan2(np.sin(angles), np.cos(angles))

def deltas_to_actions(deltas, pred_horizon, action_horizon, learn_angle=True):
    # diffusion output should be denoised action deltas of x , y and cos , sin with relate to the current state
    action_pred_deltas = deltas

    # Init action traj
    action_pred = torch.zeros_like(action_pred_deltas)

    ## Cumsum 
    action_pred[:, :, :2] = torch.cumsum(
        action_pred_deltas[:, :, :2], dim=1
    )  # convert position and orientation deltas into waypoints in local coords

    if learn_angle:
        action_pred[:, :, 2:] = F.normalize(
            action_pred_deltas[:, :, 2:].clone(), dim=-1
        )  # normalize the angle prediction to be fit with orientation representation [cos(theta), sin(theta)] >> (-1,1) normalization

    action_pred = action_pred[:,:action_horizon,:]
    
    return action_pred

    
    
    
    


    


    

def mask_target_context(lin_encoding, target_context_mask):
    # Expand target_context_mask to have the same number of features as lin_encoding
    target_context_mask_expanded = target_context_mask.unsqueeze(-1).expand_as(lin_encoding)
    
    # Mask the specific timesteps
    masked_lin_encoding = lin_encoding * (1 - target_context_mask_expanded)
    
    return masked_lin_encoding

def compute_angles_from_waypoints(waypoints):
    """
    Compute angles from waypoints represented by [cos(angle), sin(angle)] values.

    Args:
        waypoints (torch.Tensor): A tensor of shape (N, 2) where each row contains [cos(angle), sin(angle)].

    Returns:
        torch.Tensor: A tensor of computed angles in radians, shape (N,).
    """
    # waypoints is a tensor of shape (N, 2) with [cos(angle), sin(angle)] values
    cos_values = waypoints[:,:,0]  # Extract cos(angle)
    sin_values = waypoints[:,:,1]  # Extract sin(angle)
    
    # Use atan2 to compute the angles
    angles = torch.atan2(sin_values, cos_values)
    
    return angles