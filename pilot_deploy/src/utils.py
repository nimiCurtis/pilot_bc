
import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image as PILImage
from typing import List, Tuple, Dict, Optional
from pilot_train.data.data_utils import IMAGE_ASPECT_RATIO
from sensor_msgs.msg import Image
import time

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


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()

def from_numpy(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).float()

def msg_to_pil(msg: Image) -> PILImage.Image:
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, -1)
    pil_image = PILImage.fromarray(img)
    return pil_image

def pil_to_msg(pil_img: PILImage.Image, encoding="mono8") -> Image:
    img = np.asarray(pil_img)  
    ros_image = Image(encoding=encoding)
    ros_image.height, ros_image.width, _ = img.shape
    ros_image.data = img.ravel().tobytes() 
    ros_image.step = ros_image.width
    return ros_image

def transform_images(pil_imgs: List[PILImage.Image], image_size: List[int], center_crop: bool = False) -> torch.Tensor:
    """Transforms a list of PIL image to a torch tensor."""
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                    0.229, 0.224, 0.225]),   ## There is a Normazalize here be carefull if there another one
        ]
    )
    
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        
        if pil_img.mode == 'RGBA':
            pil_img = pil_img.convert('RGB')

        w, h = pil_img.size
        if center_crop:
            if w > h:
                pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))  # crop to the right ratio
            else:
                pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        pil_img = pil_img.resize(image_size) 
        transf_img = transform_type(pil_img)
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)

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

def tic():
    return time.time()


def toc(t):
    return float(tic()) - float(t)

