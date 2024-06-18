from pilot_models.policy.vint import ViNT
from pilot_models.policy.pidiff import PiDiff
from omegaconf import DictConfig

# Registry of available models
model_registry = {
    'vint': ViNT,
    'pidiff': PiDiff
    # Add new models here as you develop them, e.g., 'resnet': ResNet
}

def get_policy_model(policy_model_cfg: DictConfig,
                    encoder_model_cfg: DictConfig,
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
    model_name = policy_model_cfg.get('name')
    model_class = model_registry.get(model_name)

    if not model_class:
        raise ValueError(f"Model {model_name} is not registered or supported.")

    # Instantiate the model with the given configurations
    model = model_class(
        policy_model_cfg=policy_model_cfg,
        encoder_model_cfg=encoder_model_cfg,
        data_cfg=data_cfg
    )
    return model