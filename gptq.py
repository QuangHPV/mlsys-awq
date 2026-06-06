import math
import torch
import torch.nn as nn

# Memory-bounded GPTQ: quantize one block at a time so only its Hessian lives on GPU.


class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        self.dev = layer.weight.device
        self.rows = layer.weight.shape[0]    # C_out
        self.columns = layer.weight.shape[1] # C_in
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp):
        # inp: (batch, seq, C_in) or (tokens, C_in)
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        n = inp.shape[0]
        inp = inp.t().float()  # (C_in, tokens)
        # running H = (2/N) X X^T, with sqrt folded into inp
        self.H *= self.nsamples / (self.nsamples + n)
        self.nsamples += n
        inp = math.sqrt(2 / self.nsamples) * inp
        self.H += inp.matmul(inp.t())

    def fasterquant(self, blocksize=128, percdamp=0.01, group_size=128):
        W = self.layer.weight.data.clone().float()  # (C_out, C_in)
        C_out, C_in = W.shape
        n_groups = C_in // group_size

        H = self.H.clone()
        # columns with zero diagonal were never activated — zero them out
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        # double-Cholesky (OBQ): upper factor's diagonal gives the H^{-1}[i,i] we need
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(C_in, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        Hinv = torch.linalg.cholesky(H, upper=True)

        all_scales = torch.zeros((C_out, n_groups), device=self.dev)
        all_zeros  = torch.zeros((C_out, n_groups), device=self.dev)
        Q_int = torch.zeros((C_out, C_in), device=self.dev, dtype=torch.uint8)

        # placeholders; overwritten at the first group boundary (col 0)
        scale = torch.ones(C_out, device=self.dev)
        zero  = torch.zeros(C_out, device=self.dev)

        for i1 in range(0, C_in, blocksize):
            i2 = min(i1 + blocksize, C_in)
            W1    = W[:, i1:i2].clone()
            Err1  = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(i2 - i1):
                col = i1 + i
                w = W1[:, i]          # (C_out,)  current column, error-corrected by prev cols
                d = Hinv1[i, i]       # diagonal of H^{-1} for this column

                # recompute per-group scale/zero at each boundary from the corrected W
                if col % group_size == 0:
                    g = col // group_size
                    w_group = W[:, col:col + group_size]
                    w_min = w_group.min(dim=1).values   # (C_out,)
                    w_max = w_group.max(dim=1).values
                    scale = ((w_max - w_min) / 15).clamp(min=1e-8)
                    zero  = (-w_min / scale).round()
                    all_scales[:, g] = scale
                    all_zeros[:, g]  = zero

                q_int = (w / scale + zero).round().clamp(0, 15).to(torch.uint8)
                q     = scale * (q_int.float() - zero)   # dequantized, for error propagation
                Q_int[:, col] = q_int

                err1 = (w - q) / d
                # propagate error to remaining columns in this block
                W1[:, i:] -= err1.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
                Err1[:, i] = err1

            # propagate block error to all subsequent columns
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        # dequantized weight, written back so later blocks calibrate on its quant error
        W_dq = Q_int.reshape(C_out, n_groups, group_size).float()
        W_dq = (W_dq - all_zeros.unsqueeze(-1)) * all_scales.unsqueeze(-1)
        W_dq = W_dq.reshape(C_out, C_in)

        # pack two INT4 per byte: even column → low nibble, odd → high nibble
        w_packed = (Q_int[:, 0::2] & 0xF) | ((Q_int[:, 1::2] & 0xF) << 4)
        return w_packed, all_scales.half(), all_zeros.half(), W_dq.half()


def quantize_model_gptq(model, calib_data, group_size=128, blocksize=128, percdamp=0.01,
                        batch_size=4, n_samples=128):
    if not (hasattr(model, "model") and hasattr(model.model, "layers")):
        raise ValueError("quantize_model_gptq expects a Llama/Qwen-style model with .model.layers")

    device = next(model.parameters()).device
    layers = model.model.layers
    # dotted prefix of the decoder block list, e.g. "model.layers"
    layers_prefix = next(n for n, m in model.named_modules() if m is layers)

    # GPTQ converges with ~128 samples; cap to keep per-block full-model passes cheap
    calib = calib_data[:n_samples]

    model.eval()
    quant_params = {}
    for i, layer in enumerate(layers):
        subset = {
            f"{layers_prefix}.{i}.{rel}": m
            for rel, m in layer.named_modules()
            if isinstance(m, nn.Linear)
        }
        gptqs = {name: GPTQ(m) for name, m in subset.items()}
        handles = [
            m.register_forward_hook(lambda _, inp, __, g=gptqs[name]: g.add_batch(inp[0].detach()))
            for name, m in subset.items()
        ]

        # full-model forward with hooks only on this block's linears (sequential GPTQ)
        with torch.no_grad():
            for b in range(0, len(calib), batch_size):
                model(calib[b:b + batch_size].to(device), use_cache=False)
        for h in handles:
            h.remove()

        for name, g in gptqs.items():
            module = subset[name]
            w_packed, scales, zeros, W_dq = g.fasterquant(blocksize=blocksize, percdamp=percdamp, group_size=group_size)
            quant_params[name] = {
                "weight_int4": w_packed.cpu(),
                "scales":      scales.cpu(),
                "zeros":       zeros.cpu(),
                # no activation rescaling in GPTQ; ones make the awq_scale division a no-op
                "awq_scale":   torch.ones(module.weight.shape[1], dtype=torch.float16),
                "bias":        module.bias,
            }
            module.weight.data = W_dq.to(module.weight.dtype)  # write back for sequential GPTQ
            del g.H
        torch.cuda.empty_cache()
        print(f"quantized block {i}")

    return quant_params
