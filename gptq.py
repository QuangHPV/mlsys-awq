import math
import torch
import torch.nn as nn


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
        # running update of H = (2/N) * X X^T, absorbing sqrt into inp to avoid
        # recomputing the full product at each step
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

        # damp, then compute upper Cholesky of H^{-1}
        # double-Cholesky trick from OBQ: gives Hinv[i,i] == the exact diagonal
        # entry of H^{-1} we need for the per-column error update
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
            W1    = W[:, i1:i2].clone()   # working copy of the current block
            Err1  = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(i2 - i1):
                col = i1 + i
                w = W1[:, i]          # (C_out,)  current column, error-corrected by prev cols
                d = Hinv1[i, i]       # diagonal of H^{-1} for this column

                # recompute per-group (per-row) scale and zero at each group boundary,
                # using the cross-block-corrected W so scale tracks the running state
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

        # pack two INT4 per byte: even column → low nibble, odd → high nibble
        w_packed = (Q_int[:, 0::2] & 0xF) | ((Q_int[:, 1::2] & 0xF) << 4)
        return w_packed, all_scales.half(), all_zeros.half()


def quantize_model_gptq(model, calib_data, group_size=128, blocksize=128, percdamp=0.01, batch_size=4):
    gptq_layers = {}
    hooks = []
    module_map = {n: m for n, m in model.named_modules()}

    def make_hook(gptq_obj):
        def hook(_, inp, __):
            gptq_obj.add_batch(inp[0].detach())
        return hook

    for name, module in module_map.items():
        if not isinstance(module, nn.Linear):
            continue
        if name in ("lm_head", "model.embed_tokens"):
            continue
        g = GPTQ(module)
        gptq_layers[name] = g
        hooks.append(module.register_forward_hook(make_hook(g)))

    model.eval()
    with torch.no_grad():
        for i in range(0, len(calib_data), batch_size):
            model(calib_data[i:i + batch_size].to(next(model.parameters()).device))

    for h in hooks:
        h.remove()

    quant_params = {}
    for name, g in gptq_layers.items():
        module = module_map[name]
        w_packed, scales, zeros = g.fasterquant(blocksize=blocksize, percdamp=percdamp, group_size=group_size)
        quant_params[name] = {
            "weight_int4": w_packed.cpu(),
            "scales":      scales.cpu(),
            "zeros":       zeros.cpu(),
            # no activation rescaling in GPTQ; ones make AWQLinear a no-op on this path
            "awq_scale":   torch.ones(module.weight.shape[1], dtype=torch.float16),
            "bias":        module.bias,
        }
        del g.H
        torch.cuda.empty_cache()
        print(f"quantized {name}")

    return quant_params
