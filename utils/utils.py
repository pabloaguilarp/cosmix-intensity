import utils.models as models

def build_model(config):
    """
    Instantiate model
    Use `model = build_model(config.mode)` to instantiate model
    """
    if not hasattr(config, "name"):
        raise ValueError(f"Config name must be defined")
    if not hasattr(config, "params"):
        raise ValueError(f"Config parameters must be defined")

    params = config.params.values()
    Model = getattr(models, config.name)
    model = Model(*params)
    return model