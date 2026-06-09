import math
import torch
import torch.nn as nn


class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        self.dev = layer.weight.device
        self.rows = layer.weight.shape[0]
        self.columns = layer.weight.shape[1]
        self.H = torch.zeros((self.columns, self.columns), dtype=torch.float32, device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp):
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        inp = inp.t().float().to(self.dev)
        n = inp.shape[1]
        self.H *= self.nsamples / (self.nsamples + n)
        self.nsamples += n
        inp = math.sqrt(2.0 / self.nsamples) * inp
        self.H += inp.matmul(inp.t())

    def fasterquant(self, blocksize=128, percdamp=0.01, group_size=128):
        W = self.layer.weight.data.clone().float()
        C_out, C_in = W.shape

        assert C_in % group_size == 0, "C_in must be divisible by group_size"
        assert C_in % 2 == 0, "C_in must be even for INT4 packing"

        n_groups = C_in // group_size

        H = self.H.to(self.dev)
        del self.H

        dead = torch.diag(H) < 1e-8
        H[dead, dead] = 1
        W[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        diag_idx = torch.arange(C_in, device=self.dev)
        H[diag_idx, diag_idx] += damp

        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        del H

        all_scales = torch.zeros((C_out, n_groups), device=self.dev)
        all_zeros = torch.zeros((C_out, n_groups), device=self.dev)
        Q_int = torch.zeros((C_out, C_in), dtype=torch.uint8, device=self.dev)

        for i1 in range(0, C_in, blocksize):
            i2 = min(i1 + blocksize, C_in)
            W1 = W[:, i1:i2].clone()
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(i2 - i1):
                col = i1 + i
                w = W1[:, i]
                d = Hinv1[i, i]

                if col % group_size == 0:
                    g = col // group_size
                    w_group = W1[:, i:i + group_size]
                    w_min = w_group.min(dim=1).values
                    w_max = w_group.max(dim=1).values
                    scale = ((w_max - w_min) / 15.0).clamp(min=1e-8)
                    zero = (-w_min / scale).round().clamp(0, 15)
                    all_scales[:, g] = scale
                    all_zeros[:, g] = zero

                q_int = (w / scale + zero).round().clamp(0, 15).to(torch.uint8)
                q = scale * (q_int.float() - zero)

                Q_int[:, col] = q_int

                err1 = (w - q) / d
                W1[:, i:] -= torch.outer(err1, Hinv1[i, i:])
                Err1[:, i] = err1

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        del Hinv, W

        w_packed = (Q_int[:, 0::2] & 0xF) | ((Q_int[:, 1::2] & 0xF) << 4)
        del Q_int

        w_packed_cpu = w_packed.cpu()
        scales_cpu = all_scales.half().cpu()
        zeros_cpu = all_zeros.half().cpu()

        s = all_scales.half().unsqueeze(-1)
        z = all_zeros.half().unsqueeze(-1)
        lo = (w_packed_cpu & 0xF).to(torch.int16)
        hi = ((w_packed_cpu >> 4) & 0xF).to(torch.int16)
        q_full = torch.stack([lo, hi], dim=-1).reshape(C_out, n_groups, group_size).half()
        W_dq = ((q_full - z.cpu()) * s.cpu()).reshape(C_out, C_in)

        return w_packed_cpu, scales_cpu, zeros_cpu, W_dq


def quantize_model_gptq(model, calib_data, group_size=128, blocksize=128, percdamp=0.01,
                        batch_size=4, n_samples=128):
    if not (hasattr(model, "model") and hasattr(model.model, "layers")):
        raise ValueError("quantize_model_gptq expects a Llama/Qwen-style model with .model.layers")

    device = next(model.parameters()).device
    layers = model.model.layers
    calib = calib_data[:n_samples]

    model.eval()

    class _StopForward(Exception):
        pass

    captured = []

    def _capture_hook(module, args, kwargs):
        captured.append(
            ([a.detach().cpu() if isinstance(a, torch.Tensor) else a for a in args],
             {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
              for k, v in kwargs.items()})
        )
        raise _StopForward()

    h = layers[0].register_forward_pre_hook(_capture_hook, with_kwargs=True)
    try:
        for b in range(0, len(calib), batch_size):
            try:
                model(calib[b:b + batch_size].to(device), use_cache=False)
            except _StopForward:
                pass
    finally:
        h.remove()

    def _to_device(x):
        return x.to(device) if isinstance(x, torch.Tensor) else x

    sample_args, sample_kwargs = captured[0]
    # args[0] is hidden_states; everything else is positional context
    layer_kwargs = {k: _to_device(v) for k, v in sample_kwargs.items()}
    layer_extra_args = [_to_device(a) for a in sample_args[1:]]

    inps = torch.cat([args[0] for args, _ in captured], dim=0)

    quant_params = {}

    for layer_idx, layer in enumerate(layers):
        layer = layer.to(device)

        subset = {
            name: module
            for name, module in layer.named_modules()
            if isinstance(module, nn.Linear)
        }
        gptqs = {name: GPTQ(module) for name, module in subset.items()}

        hooks = []
        for name, module in subset.items():
            def _make_hook(g):
                def _h(mod, inp, out):
                    g.add_batch(inp[0].detach())
                return _h
            hooks.append(module.register_forward_hook(_make_hook(gptqs[name])))

        with torch.no_grad():
            outs = []
            for b in range(0, len(inps), batch_size):
                hidden = inps[b:b + batch_size].to(device)
                out = layer(hidden, *layer_extra_args, **layer_kwargs)
                outs.append((out[0] if isinstance(out, (tuple, list)) else out).cpu())

        for h in hooks:
            h.remove()

        for name, g in gptqs.items():
            module = subset[name]
            w_packed, scales, zeros, W_dq = g.fasterquant(
                blocksize=blocksize, percdamp=percdamp, group_size=group_size
            )
            full_name = f"model.layers.{layer_idx}.{name}"
            quant_params[full_name] = {
                "weight_int4": w_packed,
                "scales": scales,
                "zeros": zeros,
                "awq_scale": torch.ones(module.weight.shape[1], dtype=torch.float16),
                "bias": module.bias,
            }
            module.weight.data = W_dq.to(device=device, dtype=module.weight.dtype)
            del g

        inps = torch.cat(outs, dim=0)
        layer.cpu()
        torch.cuda.empty_cache()
        print(f"quantized block {layer_idx}/{len(layers)}")

    return quant_params