from typing import Dict, Any

def get_default_config() -> Dict[str, Any]:
    """
    Returns the default configuration for the Autoformer model.

    Returns:
        Dict[str, Any]: Default configuration dictionary
    """
    return {
        'input_dim': 10,
        'model_dim': 512,
        'num_heads': 8,
        'autocorrelation_factor': 2,
        'kernel_size': 25,
        'num_encoder_layers': 2,
        'num_decoder_layers': 1,
        'dropout_rate': 0.1,
        'use_layer_norm': False,
    }

def update_config(config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Updates the configuration dictionary with new values.

    Args:
        config (Dict[str, Any]): Original configuration dictionary
        **kwargs: New key-value pairs to update in the configuration

    Returns:
        Dict[str, Any]: Updated configuration dictionary
    """
    new_config = config.copy()
    new_config.update(kwargs)
    return new_config