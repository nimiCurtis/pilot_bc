

from omegaconf import DictConfig


def get_vision_encoder_model(vision_encoder_config, data_config):
    """
    Instantiate a model based on the provided configuration.

    Args:
    config (dict): Configuration dictionary with keys 'name', 'version', 'late_fusion', 'in_channels', and 'pretrained'

    Returns:
    An instance of the requested model.
    """
    
    from pilot_models.vision_encoder.efficientnet import EfficientNet
    # from pilot_models.vision_encoder.vit import ViT
    from pilot_models.vision_encoder.depth_feature_extractor import DepthFeatureExtractor

    # Registry of available models
    vision_model_registry = {
        'efficientnet': EfficientNet,
        # 'vit': ViT,
        'depth_feature_extractor': DepthFeatureExtractor,
        # Add new models here as you develop them, e.g., 'resnet': ResNet
    }

    model_name = vision_encoder_config.get('name')
    model_class = vision_model_registry.get(model_name)

    if not model_class:
        raise ValueError(f"Model {model_name} is not registered or supported.")

    # Instantiate the model
    model = model_class(vision_encoder_config = vision_encoder_config,
                        data_config = data_config)
    return model

def get_linear_encoder_model(linear_encoder_config, data_config):
    """
    Instantiate a model based on the provided configuration.

    Args:
    config (dict): Configuration dictionary with keys 'name', 'version', 'late_fusion', 'in_channels', and 'pretrained'

    Returns:
    An instance of the requested model.
    """
    
    from pilot_models.linear_encoder.mlp import MLP
    from pilot_models.linear_encoder.tcn import TCN
    
    # Registry of available models
    linear_model_registry = {
    'mlp': MLP,
    'tcn': TCN,
    }
    
    model_name = linear_encoder_config.get('name')
    model_class = linear_model_registry.get(model_name)

    if not model_class:
        raise ValueError(f"Model {model_name} is not registered or supported.")

    # Instantiate the model
    model = model_class(linear_encoder_config = linear_encoder_config,
                        data_config = data_config)
    return model


def get_policy_model(policy_model_cfg: DictConfig,
                    vision_encoder_model_cfg: DictConfig,
                    linear_encoder_model_cfg: DictConfig,
                    data_cfg: DictConfig):
    """
    Instantiate a policy model based on the provided configurations. The function
    retrieves the model class from a registry using the model's name specified in
    `policy_model_cfg` and initializes it with given configurations.

    Args:
        policy_model_cfg (DictConfig): Configuration for the policy model including
            model specific parameters such as 'name'.
        encoder_model_cfg (DictConfig): Configuration for the encoder model, which
            might be used by the policy model.
        training_cfg (DictConfig): Configuration related to training aspects, which
            might include learning rates, batch sizes, etc.
        data_cfg (DictConfig): Data handling configurations, potentially including
            paths, preprocessing details, and other data-related parameters.

    Returns:
        An instance of the requested model as defined in `policy_model_cfg`.

    Raises:
        ValueError: If the model specified in `policy_model_cfg` is not registered
                    or not found in the model registry.
    """
    
    from pilot_models.policy.pidiff import PiDiff
    
    # Registry of available models
    policy_model_registry = {
    'pidiff': PiDiff,
    }
    
    model_name = policy_model_cfg.get('name')
    model_class = policy_model_registry.get(model_name)

    if not model_class:
        raise ValueError(f"Model {model_name} is not registered or supported.")

    # Instantiate the model with the given configurations
    model = model_class(
        policy_model_cfg=policy_model_cfg,
        vision_encoder_model_cfg=vision_encoder_model_cfg,
        linear_encoder_model_cfg = linear_encoder_model_cfg,
        data_cfg=data_cfg
    )
    return model




