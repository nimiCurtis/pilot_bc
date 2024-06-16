import torch
import torchvision.transforms.functional as TF
from typing import List, Tuple
from pilot_utils.data.data_utils import IMAGE_ASPECT_RATIO
from PIL import Image as PILImage
from PIL import Image

def transform_images(pil_imgs: List[PILImage.Image], image_size: List[int], center_crop: bool = False) -> torch.Tensor:
    """Transforms a list of PIL image to a torch tensor."""
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        
        if pil_img.mode == 'RGBA':
            pil_img = pil_img.convert('RGB')

        # w, h = pil_img.size
        # if center_crop:
        #     if w > h:
        #         pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))  # crop to the right ratio
        #     else:
        #         pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        # pil_img = pil_img.resize(image_size) 
        transf_img = resize_and_aspect_crop(pil_img, image_resize_size=image_size)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=0)

def resize_and_aspect_crop(
    img: Image.Image, image_resize_size: Tuple[int, int], aspect_ratio: float = IMAGE_ASPECT_RATIO
):
    # w, h = img.size
    # if w > h:
    #     img = TF.center_crop(img, (h, int(h * aspect_ratio)))  # crop to the right ratio
    # else:
    #     img = TF.center_crop(img, (int(w / aspect_ratio), w))
    img = img.resize(image_resize_size)
    resize_img = TF.to_tensor(img)
    return resize_img

