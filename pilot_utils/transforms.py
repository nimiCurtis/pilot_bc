import torch
import torchvision.transforms.functional as TF
from typing import List, Tuple
from pilot_utils.data.data_utils import IMAGE_ASPECT_RATIO
from PIL import Image as PILImage
from PIL import Image
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms

# # Check vectorized operations
# def transform_images(pil_imgs: List[PILImage.Image],
#                     transform: transforms) -> torch.Tensor:
#     """Transforms a list of PIL image to a torch tensor."""
#     if type(pil_imgs) != list:
#         pil_imgs = [pil_imgs]
#     transf_imgs = []
#     for pil_img in pil_imgs:
#         if pil_img.mode == 'RGBA':
#             pil_img = pil_img.convert('RGB')
#         transf_img = transform(pil_img)
#         transf_imgs.append(transf_img)
#     return torch.cat(transf_imgs)


def transform_images(pil_imgs: List[Image.Image], transform: transforms.Compose) -> torch.Tensor:
    """Transforms a list of PIL images to a torch tensor using batch processing."""
    # Convert all images to RGB mode if necessary
    pil_imgs = [img.convert('RGB') if img.mode == 'RGBA' else img for img in pil_imgs]
    # Apply the transform to the entire batch
    transf_imgs = transform(pil_imgs)
    return torch.cat(transf_imgs)

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

        to_image = transforms.ToImage()
        to_uint8 = transforms.ToDtype(torch.uint8,scale=True)
        to_float32 = transforms.ToDtype(torch.float32,scale=True)
        resize = transforms.Resize((self.width, self.height),
                                interpolation=transforms.InterpolationMode.BILINEAR,
                                antialias=True)
        
        random_erasing = transforms.RandomErasing(p=0.1,
                                                scale=(0.02, 0.02),
                                                ratio=(1., 2.))
        random_erasing = transforms.RandomApply(transforms=[random_erasing], p=0.1)
        
        random_crop = RandomAspectCrop(aspect_ratio=self.image_aspect_ratio, offset=10)
        center_crop = AspectCenterCrop(aspect_ratio=self.image_aspect_ratio)
        
        random_mask = MaskImage(img_patch_size=16, img_masking_prob=0.015)
        random_mask = transforms.RandomApply(transforms=[random_mask], p=0.1)
        
        random_rotation = transforms.RandomRotation(degrees=3)
        random_rotation = transforms.RandomApply(transforms=[random_rotation], p=0.1)
        
        ## TODO: modify it to rgb as well
        normalize = transforms.Normalize(mean=[0.5], std=[0.5]) 

        ### TRAIN TRANSFORMS ###
        train_transform =  transforms.Compose([
                                            ## start of pipline
                                            to_image,
                                            to_uint8,
                                            resize,
                                            ## main transforms
                                            # TODO: try to run with transforms
                                            # random_erasing,
                                            # random_rotation,
                                            # random_mask,
                                            # end of pipeline
                                            to_float32,
                                            normalize
                                        ])
        
        ### EVAL TRANSFORMS ###
        eval_transform =  transforms.Compose([
                                            ## start of pipline
                                            to_image,
                                            to_uint8,
                                            resize,
                                            ## end of pipeline
                                            to_float32,
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

# Custom MaskImage transform for v2
class MaskImage(nn.Module):
    def __init__(self, img_patch_size: int, img_masking_prob: float):
        super(MaskImage, self).__init__()
        self.img_patch_size = img_patch_size
        self.img_masking_prob = img_masking_prob

    def forward(self, x):
        if x.ndim == 3 and x.shape[0] == 1:  # Ensure the input is (1, H, W)
            x = x.squeeze(0)  # Remove the channel dimension for unfolding
            img_patch = x.unfold(0, self.img_patch_size, self.img_patch_size).unfold(1, self.img_patch_size, self.img_patch_size)
            mask = torch.rand((x.shape[0] // self.img_patch_size, x.shape[1] // self.img_patch_size)) < self.img_masking_prob
            mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(img_patch)
            x = x.clone()
            x.unfold(0, self.img_patch_size, self.img_patch_size).unfold(1, self.img_patch_size, self.img_patch_size)[mask] = 0
            x = x.contiguous()
            x = x.unsqueeze(0)  # Add the channel dimension back
        return x
