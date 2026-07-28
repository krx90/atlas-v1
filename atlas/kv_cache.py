"""Cached autoregressive decoding for Kronos.

Profiling put **89% of scan time in `model.decode_s1`**, called once per
autoregressive step. To predict bar 513 the model runs the transformer over all
512 previous bars; to predict bar 514 it does it again over 513. Nothing is
reused, so a 5-step forecast pushes 5 x 25 x 512 = 64,000 token-positions
through 8 layers when only 25 tokens per step are actually new.

This module keeps the keys and values instead. The context is encoded once
(prefill), and each subsequent step processes a single new token against the
cache. Cost per step drops from O(context) to O(1), and the saving grows with
the horizon: ~5x at 5 steps, ~20x at 21, ~90x over a full intraday session.

Nothing here mutates the vendored Kronos classes. The forward pass is
reimplemented against their weights, so `vendor/` stays a clean checkout and the
uncached path remains available to check against.

Three details make this correct rather than merely fast, and each fails
*silently* if got wrong -- producing plausible numbers rather than an error:

1. **RoPE needs a position offset.** `RotaryPositionalEmbedding.forward` always
   rotates from position 0. A token arriving at position 512 must be rotated for
   512, not 0. Cached keys keep the rotation they were given when new, which is
   correct because RoPE encodes *relative* position.

2. **`is_causal` must be False when decoding.** With one query against N cached
   keys, PyTorch's causal mask aligns top-left and would let the new token attend
   only to position 0. During prefill (query and key lengths equal) it is right.

3. **The context window must not roll.** Kronos slides the buffer once the
   sequence exceeds `max_context`, which shifts every cached key's position and
   invalidates the whole cache. Callers must keep `lookback + horizon <=
   max_context`; `check_fits` enforces it rather than trusting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


class ContextTooLong(ValueError):
    """The window would roll, which a KV cache cannot survive."""


def check_fits(lookback: int, horizon: int, max_context: int) -> None:
    """Refuse a configuration whose context window would slide mid-generation."""
    if lookback + horizon > max_context:
        raise ContextTooLong(
            f"lookback {lookback} + horizon {horizon} exceeds max_context "
            f"{max_context}. The buffer would roll and every cached key's "
            f"position would shift. Use lookback <= {max_context - horizon}."
        )


@dataclass
class LayerCache:
    """Keys and values accumulated for one attention layer."""

    k: torch.Tensor | None = None
    v: torch.Tensor | None = None

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.shape[2]

    def append(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)
        return self.k, self.v


@dataclass
class Cache:
    """One `LayerCache` per transformer layer."""

    layers: list[LayerCache] = field(default_factory=list)

    @classmethod
    def empty(cls, n_layers: int) -> Cache:
        return cls([LayerCache() for _ in range(n_layers)])

    @property
    def length(self) -> int:
        return self.layers[0].length if self.layers else 0


def _rope(rotary, q: torch.Tensor, k: torch.Tensor, offset: int):
    """Apply rotary embeddings starting at absolute position `offset`.

    The upstream implementation always starts at 0; with a cache the new token
    sits at `offset`, so the cos/sin table is built to cover `offset + T` and
    sliced. `k` is the *new* keys only -- cached keys keep the rotation they were
    given when they were new, which is what makes RoPE's relative property hold.
    """
    length = offset + q.shape[-2]
    cos, sin = rotary._update_cos_sin_cache(q, length)
    cos = cos[:, :, offset:length, :]
    sin = sin[:, :, offset:length, :]
    return (
        (q * cos) + (rotary._rotate_half(q) * sin),
        (k * cos) + (rotary._rotate_half(k) * sin),
    )


def _attention(attn, x: torch.Tensor, layer_cache: LayerCache):
    """Self-attention over the new tokens plus everything cached."""
    batch, length, _ = x.shape
    offset = layer_cache.length

    def heads(proj):
        return proj(x).view(batch, length, attn.n_heads, attn.head_dim).transpose(1, 2)

    q, k, v = heads(attn.q_proj), heads(attn.k_proj), heads(attn.v_proj)
    q, k = _rope(attn.rotary, q, k, offset)
    k, v = layer_cache.append(k, v)

    # Causal only while prefilling. One query against N cached keys must see all
    # of them -- PyTorch's causal mask would align top-left and expose only the
    # first position.
    out = F.scaled_dot_product_attention(q, k, v, is_causal=length > 1)
    out = out.transpose(1, 2).contiguous().view(batch, length, attn.d_model)
    return attn.resid_dropout(attn.out_proj(out))


def decode_s1(model, s1_ids, s2_ids, stamp, cache: Cache):
    """`Kronos.decode_s1` with a KV cache. Returns (logits, context).

    Mirrors the upstream body exactly -- embedding, time embedding, token
    dropout, the transformer stack, final norm, head -- differing only in that
    attention reads and extends `cache`.
    """
    x = model.embedding([s1_ids, s2_ids])
    if stamp is not None:
        x = x + model.time_emb(stamp)
    x = model.token_drop(x)

    for layer, layer_cache in zip(model.transformer, cache.layers, strict=True):
        x = x + _attention(layer.self_attn, layer.norm1(x), layer_cache)
        x = x + layer.ffn(layer.norm2(x))

    x = model.norm(x)
    return model.head(x), x


@torch.inference_mode()
def generate(
    tokenizer,
    model,
    x: torch.Tensor,
    x_stamp: torch.Tensor,
    y_stamp: torch.Tensor,
    pred_len: int,
    *,
    max_context: int,
    clip: float = 5.0,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.9,
    sample: bool = True,
):
    """Cached equivalent of `auto_regressive_inference`, one sample per row.

    `x` is already normalised and shaped (batch, seq, features); replicate rows
    beforehand to draw multiple paths, exactly as the uncached path does.
    """
    from model.kronos import sample_from_logits  # noqa: PLC0415 -- vendored

    check_fits(x.shape[1], pred_len, max_context)

    x = torch.clip(x, -clip, clip)
    s1, s2 = tokenizer.encode(x, half=True)
    full_stamp = torch.cat([x_stamp, y_stamp], dim=1)

    cache = Cache.empty(len(model.transformer))
    context_len = s1.shape[1]

    # Prefill: the whole context in one pass, which is the only time the full
    # sequence is processed.
    logits, ctx = decode_s1(model, s1, s2, full_stamp[:, :context_len], cache)

    # `decode_s2` cross-attends with the *entire* context as key and value
    # (`DependencyAwareLayer`), so unlike decode_s1 it cannot be fed only the new
    # token. The full sequence is accumulated instead -- which is exact, because
    # a causal transformer never revises an earlier position's hidden state when
    # a later one is appended.
    history = ctx

    generated_s1, generated_s2 = [], []
    for step in range(pred_len):
        pick = sample_from_logits(
            logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p,
            sample_logits=sample,
        )
        s2_logits = model.decode_s2(history, pick)
        pick2 = sample_from_logits(
            s2_logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p,
            sample_logits=sample,
        )
        generated_s1.append(pick)
        generated_s2.append(pick2)

        if step == pred_len - 1:
            break
        # Only the new token goes through the model from here on.
        position = context_len + step
        logits, ctx = decode_s1(
            model, pick, pick2, full_stamp[:, position : position + 1], cache
        )
        history = torch.cat([history, ctx], dim=1)

    out_s1 = torch.cat(generated_s1, dim=1)
    out_s2 = torch.cat(generated_s2, dim=1)
    full = [torch.cat([s1, out_s1], dim=1), torch.cat([s2, out_s2], dim=1)]
    z = tokenizer.decode(full, half=True)
    return z[:, -pred_len:, :]
