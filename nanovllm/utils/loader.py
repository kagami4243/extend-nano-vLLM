import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    should_load_weight = getattr(model, "should_load_weight", None)
    custom_weight_loader = getattr(model, "load_weight", None)

    def get_parameter(name: str):
        try:
            return model.get_parameter(name)
        except AttributeError:
            return None

    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                if (
                    should_load_weight is not None
                    and not should_load_weight(weight_name)
                ):
                    continue
                loaded_weight = None
                if custom_weight_loader is not None:
                    loaded_weight = f.get_tensor(weight_name)
                    if custom_weight_loader(weight_name, loaded_weight):
                        continue
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = get_parameter(param_name)
                        if param is None:
                            break
                        weight_loader = getattr(param, "weight_loader")
                        loaded_weight = (
                            loaded_weight
                            if loaded_weight is not None
                            else f.get_tensor(weight_name)
                        )
                        weight_loader(param, loaded_weight, shard_id)
                        break
                else:
                    param = get_parameter(weight_name)
                    if (
                        param is None
                        and weight_name == "model.embed_tokens.weight"
                        and getattr(model, "tie_word_embeddings", False)
                    ):
                        param = get_parameter("lm_head.weight")
                    if param is None:
                        continue
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    loaded_weight = (
                        loaded_weight
                        if loaded_weight is not None
                        else f.get_tensor(weight_name)
                    )
                    weight_loader(param, loaded_weight)
