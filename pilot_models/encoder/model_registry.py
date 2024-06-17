from pilot_models.encoder.efficientnet import EfficientNet
# from pilot_models.encoder.dino import Dino

from omegaconf import DictConfig, OmegaConf

# Registry of available models
model_registry = {
    'efficientnet': EfficientNet,
    # Add new models here as you develop them, e.g., 'resnet': ResNet
    # 'dino': Dino
}

def get_vision_encoder_model(config):
    """
    Instantiate a model based on the provided configuration.

    Args:
    config (dict): Configuration dictionary with keys 'name', 'version', 'late_fusion', 'in_channels', and 'pretrained'

    Returns:
    An instance of the requested model.
    """

    model_name = config.get('name')
    model_class = model_registry.get(model_name)

    if not model_class:
        raise ValueError(f"Model {model_name} is not registered or supported.")

    # Extract other parameters, providing default values where appropriate
    version = config.get('version')
    in_channels = config.get('in_channels')
    pretrained = config.get('pretrained')

    # Instantiate the model
    model = model_class(version=version, in_channels=in_channels, pretrained=pretrained)
    return model