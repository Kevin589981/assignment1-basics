from __future__ import annotations

import math

import torch
from torch import Tensor
from torch import nn


def linear(x: Tensor, weight: Tensor) -> Tensor:
    return x @ weight.transpose(-1, -2)


def embedding(token_ids: Tensor, weight: Tensor) -> Tensor:
    return weight[token_ids]


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def softmax(x: Tensor, dim: int) -> Tensor:
    shifted = x - torch.max(x, dim=dim, keepdim=True).values
    exp = torch.exp(shifted)
    return exp / torch.sum(exp, dim=dim, keepdim=True)


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    shifted = inputs - torch.max(inputs, dim=-1, keepdim=True).values
    logsumexp = torch.log(torch.sum(torch.exp(shifted), dim=-1)) + torch.max(inputs, dim=-1).values
    correct = inputs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return torch.mean(logsumexp - correct)


def rmsnorm(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    original_dtype = x.dtype
    x_float = x.to(torch.float32)
    normed = x_float * torch.rsqrt(torch.mean(x_float * x_float, dim=-1, keepdim=True) + eps)
    return (normed.to(original_dtype) * weight).to(original_dtype)


def swiglu(x: Tensor, w1: Tensor, w2: Tensor, w3: Tensor) -> Tensor:
    return linear(silu(linear(x, w1)) * linear(x, w3), w2)


def rope(x: Tensor, theta: float, max_seq_len: int, token_positions: Tensor) -> Tensor:
    del max_seq_len
    d_k = x.shape[-1]
    if d_k % 2 != 0:
        raise ValueError("RoPE requires an even embedding dimension")
    device = x.device
    dtype = x.dtype
    inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k))
    positions = token_positions.to(device=device)
    angles = positions.to(torch.float32).unsqueeze(-1) * inv_freq
    cos = torch.cos(angles).to(dtype)
    sin = torch.sin(angles).to(dtype)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None) -> Tensor:
    scores = Q @ K.transpose(-1, -2) / math.sqrt(Q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    return softmax(scores, dim=-1) @ V


def multihead_self_attention(
    x: Tensor,
    num_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    max_seq_len: int | None = None,
    theta: float | None = None,
    token_positions: Tensor | None = None,
) -> Tensor:
    *leading, seq_len, d_model = x.shape
    d_head = d_model // num_heads
    q = linear(x, q_proj_weight).reshape(*leading, seq_len, num_heads, d_head).transpose(-2, -3)
    k = linear(x, k_proj_weight).reshape(*leading, seq_len, num_heads, d_head).transpose(-2, -3)
    v = linear(x, v_proj_weight).reshape(*leading, seq_len, num_heads, d_head).transpose(-2, -3)
    if theta is not None:
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
            if leading:
                token_positions = token_positions.expand(*leading, seq_len)
        q = rope(q, theta, max_seq_len or seq_len, token_positions.unsqueeze(-2))
        k = rope(k, theta, max_seq_len or seq_len, token_positions.unsqueeze(-2))
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
    attn = scaled_dot_product_attention(q, k, v, causal)
    attn = attn.transpose(-2, -3).reshape(*leading, seq_len, d_model)
    return linear(attn, o_proj_weight)


def transformer_block(
    x: Tensor,
    weights: dict[str, Tensor],
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
) -> Tensor:
    del d_ff
    z = x + multihead_self_attention(
        rmsnorm(x, weights["ln1.weight"]),
        num_heads,
        weights["attn.q_proj.weight"],
        weights["attn.k_proj.weight"],
        weights["attn.v_proj.weight"],
        weights["attn.output_proj.weight"],
        max_seq_len=max_seq_len,
        theta=theta,
    )
    return z + swiglu(
        rmsnorm(z, weights["ln2.weight"]),
        weights["ffn.w1.weight"],
        weights["ffn.w2.weight"],
        weights["ffn.w3.weight"],
    )


def transformer_lm(
    in_indices: Tensor,
    weights: dict[str, Tensor],
    context_length: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
) -> Tensor:
    x = embedding(in_indices, weights["token_embeddings.weight"])
    for layer_idx in range(num_layers):
        prefix = f"layers.{layer_idx}."
        block_weights = {k.removeprefix(prefix): v for k, v in weights.items() if k.startswith(prefix)}
        x = transformer_block(x, block_weights, num_heads, d_ff, context_length, rope_theta)
    x = rmsnorm(x, weights["ln_final.weight"])
    return linear(x, weights["lm_head.weight"])


def gradient_clipping(parameters, max_l2_norm: float) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    total_norm = torch.sqrt(sum(torch.sum(g.detach() ** 2) for g in grads))
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for grad in grads:
            grad.mul_(scale)


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"].item()
                p.mul_(1 - lr * weight_decay)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                p.addcdiv_(exp_avg, torch.sqrt(exp_avg_sq).add_(eps), value=-step_size)
        return loss


def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it > cosine_cycle_iters:
        return min_learning_rate
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (
        max_learning_rate - min_learning_rate
    )
