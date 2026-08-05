"""Numerical check that the paper's dependent variables survive a
function-preserving reparameterisation of the residual stream, and that the
matrix diagnostics we refuse to use do not.

The reparameterisation. Write $o_\\ell$ for the output of block $\\ell$, with
$o_{-1}$ the embedding, so
$o_\\ell = o_{\\ell-1} + F_\\ell(\\mathrm{norm}(o_{\\ell-1}))$.
Pick any positive $a_\\ell$ and build a network whose residual stream is
$\\tilde o_\\ell = a_\\ell o_\\ell$:

* scale the embedding by $a_{-1}$;
* scale block $\\ell$'s output projections by $a_{\\ell-1}$;
* multiply block $\\ell$'s output by $a_\\ell / a_{\\ell-1}$ --- a scalar on the
  residual path, which real architectures carry.

Because a pre-norm block reads through a scale-free normaliser,
$\\mathrm{norm}(\\tilde o_{\\ell-1}) = \\mathrm{norm}(o_{\\ell-1})$, so every block
computes the same function of the same argument and
$\\mathrm{logits} = W_U\\,\\mathrm{norm}(\\tilde o_{L-1})$ is unchanged. Nothing is
approximated: this is a different network computing the same function.

Why the gauge must vary with depth. Under a *global* rescale $a_\\ell \\equiv c$
both $o_\\ell$ and $o_{L-1}$ scale by $c$, so
$J_\\ell = \\partial o_{L-1} / \\partial o_\\ell$ is **unchanged** and such a gauge
makes no point at all. Under a depth-varying one,
$\\tilde J_\\ell = (a_{L-1}/a_\\ell)\\, J_\\ell$, and that factor is exactly what the
matrix diagnostics pick up.

So with a step gauge --- $a_\\ell = 1$ below layer $k$ and $a_\\ell = s$ at or above
it --- three things must hold, and each is measured here rather than asserted:

1. **The logits do not move.** Same function, different weights.
2. **$R_z$ does not move**, at any band layer. The readout
   $W_U\\,\\mathrm{norm}(J_\\ell h_\\ell)$ is invariant because
   $\\tilde J_\\ell \\tilde h_\\ell = a_{L-1} J_\\ell h_\\ell$ and the norm removes the
   scalar.
3. **The mean diagonal of $J_\\ell$ does move**, by $s$ below $k$ and by $1$ at or
   above it. Measured by Hutchinson's estimator on central-difference
   Jacobian-vector products in both parameterisations. Every diagnostic
   homogeneous of nonzero degree in $J$ --- mean diagonal, distance to the
   identity, fraction of diagonal entries above a threshold --- therefore reports
   a coordinate choice.

Run in **fp32**. In bf16 the same construction drifts by ${\\sim}3\\%$ over 64
layers purely from accumulation, which is enough to blur an exact identity into a
small residual and prove nothing.

"""

from __future__ import annotations

import torch

from innerj import config
from innerj.analysis.readout import percentile_rank
from innerj.patch import attention_projection

OUT = config.DATA_ROOT / "gauge"


def step_gauge(n_layers: int, step_layer: int, scale: float) -> list[float]:
    """``a[l + 1] = a_l``, with ``a[0] = a_{-1}`` for the embedding.

    A step rather than a smooth profile because the prediction is then legible:
    the diagnostic must move by exactly ``scale`` below ``step_layer`` and by
    exactly 1 at or above it, in the same run.
    """
    return [1.0] + [scale if layer >= step_layer else 1.0 for layer in range(n_layers)]


@torch.no_grad()
def apply_gauge(model, a: list[float]) -> list:
    """Reparameterise in place. Returns the hook handles keeping it alive."""
    model._hf_model.get_input_embeddings().weight.mul_(a[0])
    handles = []
    for layer in range(model.n_layers):
        projection, _ = attention_projection(model, layer)
        for module in (projection, model.layers[layer].mlp.down_proj):
            module.weight.mul_(a[layer])
            if module.bias is not None:
                module.bias.mul_(a[layer])
        gain = a[layer + 1] / a[layer]
        if gain != 1.0:
            handles.append(
                model.layers[layer].register_forward_hook(
                    lambda _m, _args, out, gain=gain: (
                        (out[0] * gain, *out[1:]) if isinstance(out, tuple)
                        else out * gain
                    )
                )
            )
    return handles


@torch.no_grad()
def _stream(model, input_ids, layers: list[int]) -> dict[int, torch.Tensor]:
    """Residual stream at the query position, at each requested block output."""
    recorded: dict[int, torch.Tensor] = {}

    def keep(_module, _args, output, layer):
        tensor = output[0] if isinstance(output, tuple) else output
        recorded[layer] = tensor[0, -1:].detach().float()

    handles = [
        model.layers[layer].register_forward_hook(
            lambda m, a, o, layer=layer: keep(m, a, o, layer)
        )
        for layer in sorted(set(layers) | {model.n_layers - 1})
    ]
    try:
        model.forward(input_ids)
    finally:
        for handle in handles:
            handle.remove()
    return recorded


@torch.no_grad()
def read(model, lens, prompt: str, gold_token_id: int, layers: list[int]) -> dict:
    """Model logits and per-layer ``R_z`` at the query position, one forward pass."""
    input_ids = model.encode(prompt, max_length=512)
    recorded = _stream(model, input_ids, layers)
    return {
        "logits": model.unembed(recorded[model.n_layers - 1]).float().cpu()[0],
        "r_z": {
            layer: percentile_rank(
                model.unembed(lens.transport(recorded[layer], layer)).float().cpu()[0],
                gold_token_id,
            )
            for layer in layers
        },
    }


@torch.no_grad()
def mean_diagonal(model, prompt: str, layer: int, *, probes: int, epsilon: float,
                  seed: int) -> float:
    """``mean_i J[i, i]`` by Hutchinson on central-difference JVPs.

    ``E_v[v^T J v] / d`` is the mean diagonal for Rademacher ``v``, and ``J v`` is
    a symmetric difference of the final stream against a perturbation of the
    stream at ``layer``. Central rather than forward differences so the estimate
    is second-order accurate and the same ``epsilon`` is usable in both
    parameterisations, where the stream magnitudes differ.
    """
    input_ids = model.encode(prompt, max_length=512)
    final_layer = model.n_layers - 1
    device = model.layers[layer].mlp.down_proj.weight.device
    d_model = int(model._hf_model.config.get_text_config().hidden_size)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    delta = torch.zeros(d_model, device=device, dtype=torch.float32)

    def perturb(_module, _args, output):
        tensor = output[0] if isinstance(output, tuple) else output
        tensor = tensor.clone()
        tensor[0, -1] = tensor[0, -1] + delta.to(tensor.dtype)
        return (tensor, *output[1:]) if isinstance(output, tuple) else tensor

    final: dict[str, torch.Tensor] = {}

    def run() -> torch.Tensor:
        handles = [
            model.layers[layer].register_forward_hook(perturb),
            model.layers[final_layer].register_forward_hook(
                lambda _m, _a, out: final.__setitem__(
                    "h", (out[0] if isinstance(out, tuple) else out)[0, -1]
                    .detach().float()
                )
            ),
        ]
        try:
            model.forward(input_ids)
        finally:
            for handle in handles:
                handle.remove()
        return final["h"]

    total = 0.0
    for _ in range(probes):
        v = (
            torch.randint(0, 2, (d_model,), generator=generator).float() * 2 - 1
        ).to(device)
        delta.copy_(epsilon * v)
        plus = run()
        delta.copy_(-epsilon * v)
        minus = run()
        delta.zero_()
        total += float(torch.dot((plus - minus) / (2 * epsilon), v)) / d_model
    return total / probes
