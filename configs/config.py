import yaml
import inspect


def get_config(name):
    stream = open(name, 'r')
    config_dict = yaml.safe_load(stream)
    return Config(config_dict)


def get_config_dict(name):
    stream = open(name, 'r')
    config_dict = yaml.safe_load(stream)
    return config_dict


class Config:
    def __init__(self, in_dict: dict):
        assert isinstance(in_dict, dict)
        self.in_dict = in_dict
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(self, key, [Config(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key, Config(val) if isinstance(val, dict) else val)

    def __getitem__(self, key):
        return self.in_dict[key]

    def values(self):
        ignore = ["in_dict"]
        return [v for k, v in vars(self).items() if k not in ignore]

def get_init_params(cls):
    """Get the parameters of the __init__ method of a class."""
    # Get the __init__ method
    init_method = getattr(cls, "__init__", None)
    if not init_method:
        return []

    # Use inspect to get the signature of __init__
    signature = inspect.signature(init_method)
    # Extract parameter names, excluding 'self'
    params = [param.name for param in signature.parameters.values() if param.name != "self"]
    return params