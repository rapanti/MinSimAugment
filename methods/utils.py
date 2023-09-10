import torch


def get_embed_dim(model: torch.nn.Module) -> int:
    if hasattr(model, "fc"):
        embed_dim = model.fc.in_features
        model.fc = torch.nn.Identity()
    elif hasattr(model, "head"):
        embed_dim = model.head.in_features
        model.head = torch.nn.Identity()
    else:
        raise ValueError("Model must have `fc` or `head` attribute.")
    return embed_dim
