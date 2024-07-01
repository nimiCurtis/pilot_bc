from pilot_models.linear_encoder.mlp import MLP
from omegaconf import DictConfig, OmegaConf

# Registry of available models
model_registry = {
    'mlp': MLP,
    # Add new models here as you develop them, e.g., 'resnet': ResNet
    # 'dino': Dino
}

def get_linear_encoder_model(linear_encoder_config, data_config):
    """
    Instantiate a model based on the provided configuration.

    Args:
    config (dict): Configuration dictionary with keys 'name', 'version', 'late_fusion', 'in_channels', and 'pretrained'

    Returns:
    An instance of the requested model.
    """

    model_name = linear_encoder_config.get('name')
    model_class = model_registry.get(model_name)

    if not model_class:
        raise ValueError(f"Model {model_name} is not registered or supported.")

    # Instantiate the model
    model = model_class(linear_encoder_config = linear_encoder_config,
                        data_config = data_config)
    return model