from pilot_models.encoder.efficientnet import EfficientNet
from pilot_models.encoder.vit import ViT
# from pilot_models.encoder.dino import Dino

from omegaconf import DictConfig, OmegaConf

# Registry of available models
model_registry = {
    'efficientnet': EfficientNet,
    'vit': ViT,
    # Add new models here as you develop them, e.g., 'resnet': ResNet
    # 'dino': Dino
}

def get_vision_encoder_model(vision_encoder_config, data_config):
    """
    Instantiate a model based on the provided configuration.

    Args:
    config (dict): Configuration dictionary with keys 'name', 'version', 'late_fusion', 'in_channels', and 'pretrained'

    Returns:
    An instance of the requested model.
    """

    model_name = vision_encoder_config.get('name')
    model_class = model_registry.get(model_name)

    if not model_class:
        raise ValueError(f"Model {model_name} is not registered or supported.")

    # Instantiate the model
    model = model_class(vision_encoder_config = vision_encoder_config,
                        data_config = data_config)
    return model