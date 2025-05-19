import torch
from collections.abc import Sequence


def concat_features(data, keys = None):
    if keys is not None:
        if isinstance(keys, str):
            keys = [keys]
        assert isinstance(data, dict)
        assert isinstance(keys, Sequence)
        return torch.cat([data[key] for key in keys], dim=1)
    else:
        assert isinstance(data, Sequence)
        return torch.cat(data, dim=1)