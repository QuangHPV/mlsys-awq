import torch
import torch.nn as nn
from triton_kernel import dequant_gemm


class AWQLinear(nn.Module):
    def __init__(self, w_int4, scales, zeros, awq_scale, bias, group_size=128):
        super().__init__()
        self.group_size = group_size
        self.register_buffer("w_int4", w_int4)
        self.register_buffer("scales", scales)
        self.register_buffer("zeros", zeros)
        # dividing x by awq_scale inverts the per-channel scaling baked into weights
        self.register_buffer("awq_scale", awq_scale)
        self.bias = nn.Parameter(bias) if bias is not None else None

    def forward(self, x):
        orig_shape = x.shape
        x = x.view(-1, orig_shape[-1]).half()
        x = x / self.awq_scale.unsqueeze(0)
        out = dequant_gemm(x, self.w_int4, self.scales, self.zeros, self.group_size)
        if self.bias is not None:
            out = out + self.bias
        return out.view(*orig_shape[:-1], -1)


def replace_with_awq_linear(model, quant_params, group_size=128):
    for name, params in quant_params.items():
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], AWQLinear(
            w_int4=params["weight_int4"],
            scales=params["scales"],
            zeros=params["zeros"],
            awq_scale=params["awq_scale"],
            bias=params["bias"],
            group_size=group_size,
        ))
    return model