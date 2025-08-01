import torch


def autocast(enabled: bool):
    return torch.cuda.amp.autocast(enabled)


def convert_optimizer_state_dict_to_fp16(state_dict):
    for state in state_dict["state"].values():
        for k, v in state.items():
            if k != "step" and isinstance(v, torch.Tensor) and v.dtype is torch.float32:
                state[k] = v.half()

    return state_dict
