import torch

def extract_embedding(outputs, strategy="last_hidden", layer_outputs=None):
    if strategy == "first":
        return layer_outputs[0]
    elif strategy == "last_hidden":
        return outputs.last_hidden_state
    elif strategy == "second_to_last":
        return layer_outputs[-2]
    elif strategy == "sum_all":
        return torch.stack(layer_outputs, dim=0).sum(dim=0)
    elif strategy == "sum_last_four":
        return torch.stack(layer_outputs[-4:], dim=0).sum(dim=0)
    elif strategy == "concat_last_four":
        return torch.cat(layer_outputs[-4:], dim=-1)
    else:
        raise ValueError("Unknown embedding strategy")
