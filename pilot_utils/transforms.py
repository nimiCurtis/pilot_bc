import torch
import torchvision.transforms.functional as TF
from typing import List, Tuple
from pilot_utils.data.data_utils import IMAGE_ASPECT_RATIO
from PIL import Image as PILImage
from PIL import Image
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms

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

# def transform_images(pil_imgs: List[Image.Image], transform: transforms.Compose) -> torch.Tensor:
#     """Transforms a list of PIL images to a torch tensor using batch processing."""
#     # Convert all images to RGB mode if necessary
#     pil_imgs = [img.convert('RGB') if img.mode == 'RGBA' else img for img in pil_imgs]
#     # Apply the transform to each image individually
#     transf_imgs = [transform(img) for img in pil_imgs]
#     return torch.stack(transf_imgs)

# def transform_images(pil_imgs: List[Image.Image], transform: transforms.Compose) -> torch.Tensor:
#     """Transforms a list of PIL images to a torch tensor using batch processing."""
#     # Convert all images to RGB mode if necessary
#     pil_imgs = [img.convert('RGB') if img.mode == 'RGBA' else img for img in pil_imgs]
#     # Apply the transform to the entire batch
#     transf_imgs = transform(pil_imgs)
#     return torch.cat(transf_imgs)

# Define a custom data_cfg
class CustomDataConfig:
    def __init__(self, image_size: Tuple[int, int], img_type: str):
        self.image_size = image_size
        self.img_type = img_type




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

        random_erasing = transforms.RandomErasing(p=0.8,
                                                scale=(0.02, 0.02),
                                                ratio=(1., 2.))
        
        # random_crop = RandomAspectCrop(aspect_ratio=self.image_aspect_ratio, offset=10)
        # center_crop = AspectCenterCrop(aspect_ratio=self.image_aspect_ratio)
        
        random_mask = MaskImage(img_patch_size=8, img_masking_prob=0.1)
        random_mask = transforms.RandomApply(transforms=[random_mask], p=0.1)
        
        random_rotation = transforms.RandomRotation(degrees=20)
        random_rotation = transforms.RandomApply(transforms=[random_rotation], p=0.5)
        
        ## TODO: modify it to rgb as well
        normalize = transforms.Normalize(mean=[0.5], std=[0.5]) 
        
        random_crop = transforms.RandomResizedCrop(size=(self.width, self.height),
                                interpolation=transforms.InterpolationMode.BILINEAR,
                                # ratio=self.image_aspect_ratio,
                                antialias=True)
        random_crop = transforms.RandomApply(transforms=[random_crop], p=0.2)
        
        ### TRAIN TRANSFORMS ###
        train_transform =  transforms.Compose([
                                            ## start of pipline
                                            to_image,
                                            to_uint8,
                                            random_crop,
                                            resize,
                                            ## main transforms
                                            # TODO: try to run with transforms
                                            # random_erasing,
                                            random_rotation,
                                            random_mask,
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


# Function to display the original and transformed images
# Function to display the original and multiple transformed images
def show_transformed_images(image_path, transform, num_transforms=10):
    # Load the image
    img = PILImage.open(image_path)

    # Plot the original image
    fig, axes = plt.subplots(1, num_transforms + 1, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Apply the transformation multiple times and display each transformed image
    for i in range(num_transforms):
        transformed_img = transform(img)

        # Convert transformed image back to PIL Image for visualization
        transformed_img = transformed_img.permute(1, 2, 0).numpy()
        # transformed_img = np.clip(transformed_img, 0, 255).astype(np.uint8)

        axes[i + 1].imshow(transformed_img)
        axes[i + 1].set_title(f'Transform {i+1}')
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    # Define a sample image path
    image_path = '/home/roblab20/dev/pilot/pilot_bc/pilot_dataset/pilot_target_tracking/nimrod_bag-2024-06-30-17-11-31-data/visual_data/depth/3.jpg'
    # Instantiate the custom data configuration
    data_cfg = CustomDataConfig(image_size=(96, 96), img_type="depth")
    # Define a simple transform for demonstration purposes
    transform = ObservationTransform(data_cfg=data_cfg).get_transform(type="train")

    # Call the function to display the images
    show_transformed_images(image_path, transform)