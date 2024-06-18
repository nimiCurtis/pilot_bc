import torch
import torchvision.transforms.functional as TF
from typing import List, Tuple
from pilot_utils.data.data_utils import IMAGE_ASPECT_RATIO
from PIL import Image as PILImage
from PIL import Image
import random

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import Callable

# Check vectorized operations
def transform_images(pil_imgs: List[PILImage.Image],
                    transform: transforms) -> torch.Tensor:
    """Transforms a list of PIL image to a torch tensor."""
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        if pil_img.mode == 'RGBA':
            pil_img = pil_img.convert('RGB')
        transf_img = transform(pil_img)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs)

# obs_image = torch.cat([
        #         self.transform(self._load_image(f, t)) for f, t in context
        #     ])

def resize_and_aspect_crop(
    img: Image.Image, image_resize_size: Tuple[int, int], aspect_ratio: float = IMAGE_ASPECT_RATIO
):
    img = img.resize(image_resize_size)
    resize_img = TF.to_tensor(img)
    return resize_img




class ObservationTransform:
    def __init__(self, data_cfg,
                img_patch_size=8, img_gaussian_noise=0.0, img_masking_prob=0.0):
        self.width = data_cfg.image_size[0]
        self.height = data_cfg.image_size[1]
        self.channels = 1 if data_cfg.img_type == "depth" else 3
        self.img_patch_size = img_patch_size
        self.img_gaussian_noise = img_gaussian_noise
        self.img_masking_prob = img_masking_prob
        self.image_aspect_ratio = IMAGE_ASPECT_RATIO
        train_transform, eval_transform = self._define_transforms()
        self.transforms = {"train": train_transform,
                        "test": eval_transform}

    def _define_transforms(self):
        
        random_crop = RandomAspectCrop(aspect_ratio=self.image_aspect_ratio, offset=10)
        center_crop = AspectCenterCrop(aspect_ratio=self.image_aspect_ratio)
        resize = transforms.Resize((self.width, self.height), interpolation=transforms.InterpolationMode.BILINEAR)
        
        #gaussian_noise = GaussianNoise(self.img_gaussian_noise)
        #patch_masks = MaskImage(img_patch_size=16, img_masking_prob=0.3)
        
        to_tensor = transforms.ToTensor()
        
        ## TODO: modify it to rgb as well
        normalize = transforms.Normalize(mean=[0.5], std=[0.5]) 
        ### TRAIN TRANSFORMS ###
        train_transform =  transforms.Compose([random_crop,
                                            resize,
                                            to_tensor,
                                            normalize
                                        ])
        
        ### EVAL TRANSFORMS ###
        eval_transform =  transforms.Compose([center_crop,
                                                    resize,
                                                    to_tensor,
                                                    normalize
                                                ])

        return train_transform, eval_transform

    def get_transform(self, type):
        
        return self.transforms[type]

class AspectCenterCrop:
    def __init__(self, aspect_ratio: float = IMAGE_ASPECT_RATIO):
        self.aspect_ratio = aspect_ratio

    def __call__(self, img: Image.Image):
        w, h = img.size
        if w > h:
            img = TF.center_crop(img, (h, int(h * self.aspect_ratio)))  # Crop to the right ratio
        else:
            img = TF.center_crop(img, (int(w / self.aspect_ratio), w))
        return img

class RandomAspectCrop:
    def __init__(self, aspect_ratio: float = 1.0, offset: int = 10):
        self.aspect_ratio = aspect_ratio
        self.offset = offset

    def __call__(self, img: Image.Image):
        w, h = img.size
        if w > h:
            crop_height = h
            crop_width = int(h * self.aspect_ratio)
        else:
            crop_width = w
            crop_height = int(w / self.aspect_ratio)
        
        # Calculate the center
        center_x, center_y = w // 2, h // 2
        
        # Apply random offset relate to the x
        offset_x = random.randint(-self.offset, self.offset)
        offset_y = random.randint(-self.offset, self.offset)
        
        # Calculate new crop box
        left = max(0, center_x + offset_x - crop_width // 2)
        # top = max(0, center_y + offset_y - crop_height // 2)
        right = min(w, left + crop_width)
        # bottom = min(h, top + crop_height)
        
        # Currently crop only with translation from the center
        top = 0
        bottom = top+crop_height
        img = img.crop((left, top, right, bottom))
        return img

###### NOT IN USE ##########
class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, img):
        noise = torch.randn(img.size()) * self.stddev
        return img + noise

class MaskImage:
    def __init__(self, img_patch_size: int, img_masking_prob: float):
        self.img_patch_size = img_patch_size
        self.img_masking_prob = img_masking_prob

    def __call__(self, x):
        img_patch = x.unfold(1, self.img_patch_size, self.img_patch_size).unfold(2, self.img_patch_size, self.img_patch_size)
        mask = torch.rand((x.shape[0], x.shape[1] // self.img_patch_size, x.shape[2] // self.img_patch_size)) < self.img_masking_prob
        mask = mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand_as(img_patch)
        x = x.clone()
        x.unfold(1, self.img_patch_size, self.img_patch_size).unfold(2, self.img_patch_size, self.img_patch_size)[mask] = 0
        return x.contiguous()