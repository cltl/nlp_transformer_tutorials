import torch as t
import torch.nn as nn
import sys
from dataclasses import dataclass
from pathlib import Path
from functools import partial

exercises_dir = Path(__file__).parent.parent
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))


# ---------------------------------------------------------------------------
# Internal reference implementations used only by the tests below.
# These are intentionally minimal — just enough to produce ground-truth outputs.
# Prefixed with _ to signal they are test-internal and not part of the exercise.
# ---------------------------------------------------------------------------

_device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")


@dataclass
class _Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12


class _LayerNorm(nn.Module):
    """Reference LayerNorm used in test_layer_norm_epsilon."""

    def __init__(self, cfg: _Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(self, residual):
        mean = residual.mean(dim=-1, keepdim=True)
        std = (residual.var(dim=-1, keepdim=True, unbiased=False) + self.cfg.layer_norm_eps).sqrt()
        return (residual - mean) / std * self.w + self.b


class _CausalMaskReference(nn.Module):
    """Reference causal mask used in test_causal_mask.
    Provides the IGNORE buffer and a known-correct apply_causal_mask."""

    def __init__(self, cfg: _Config):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("IGNORE", t.tensor(float("-inf"), dtype=t.float32))

    def apply_causal_mask(self, attn_scores):
        all_ones = t.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device)
        mask = t.triu(all_ones, diagonal=1).bool()
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores


# ---------------------------------------------------------------------------
# Public test functions
# ---------------------------------------------------------------------------


def rand_float_test(cls, shape):
    cfg = _Config(debug=True)
    layer = cls(cfg).to(_device)
    random_input = t.randn(shape).to(_device)
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    if isinstance(output, tuple):
        output = output[0]
    print("Output shape:", output.shape, "\n")


def rand_int_test(cls, shape):
    cfg = _Config(debug=True)
    layer = cls(cfg).to(_device)
    random_input = t.randint(100, 1000, shape).to(_device)
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    if isinstance(output, tuple):
        output = output[0]
    print("Output shape:", output.shape, "\n")


def load_gpt2_test(cls, gpt2_layer, input):
    cfg = _Config(debug=True)
    layer = cls(cfg).to(_device)
    layer.load_state_dict(gpt2_layer.state_dict(), strict=False)
    print("Input shape:", input.shape)
    orig_input = input.clone()
    output = layer(orig_input)
    assert t.allclose(input, orig_input), "Input has been modified, make sure operations are not done in place"
    if isinstance(output, tuple):
        output = output[0]
    print("Output shape:", output.shape)
    try:
        reference_output = gpt2_layer(input)
    except:
        reference_output = gpt2_layer(input, input, input)
    print("Reference output shape:", reference_output.shape, "\n")
    comparison = t.isclose(output, reference_output, atol=1e-4, rtol=1e-3)
    print(f"{comparison.sum() / comparison.numel():.2%} of the values are correct\n")
    assert 1 - (comparison.sum() / comparison.numel()) < 1e-5, "More than 0.01% of the values are incorrect"


def test_layer_norm_epsilon(layer_norm, cache):
    cfg = _Config(layer_norm_eps=0.1)

    # dont worry about different weights at initialisation
    # gamma init to 1, beta init to 0
    # .to(_device) is required so that learned parameters (w, b) are on
    # the same device as the input cache — without this the test fails on MPS (Apple Silicon)
    expected = _LayerNorm(cfg).to(_device)(cache.clone())
    actual = layer_norm(cfg).to(_device)(cache.clone())
    t.testing.assert_close(expected, actual, msg="LayerNorm: Is your epsilon inside the sqrt?")


def test_causal_mask(apply_causal_mask):
    cfg = _Config()
    attn_scores = t.randn((1, 1, 5, 5)).to(_device)  # (batch, n_heads, query_pos, key_pos)
    ref = _CausalMaskReference(cfg).to(_device)

    sol_apply_causal_mask = ref.apply_causal_mask
    apply_causal_mask = partial(apply_causal_mask, self=ref)

    expected = sol_apply_causal_mask(attn_scores=attn_scores.clone())
    actual = apply_causal_mask(attn_scores=attn_scores.clone())

    def print_scores():
        print(f"Expected Attention Scores: \n {expected}")
        print(f"Actual Attention Scores: \n {actual}")
        print(f"Expected Attention Probs: \n {t.softmax(expected, dim=-1)}")
        print(f"Actual Attention Probs: \n {t.softmax(actual, dim=-1)}")

    if t.any(t.isnan(actual)):
        nan_freq = t.sum(t.isnan(actual)).item() / actual.numel()
        print_scores()
        raise ValueError(
            f"Your masked attention scores contains {nan_freq * 100}% NaNs. Make sure you aren't multiplying 0 * neg inf anywhere."
        )

    attn_probs = t.softmax(
        actual, dim=-1
    )  # ignoring the scale factor, we just want to check if the mask is applied correctly

    if t.any(t.isnan(attn_probs)):
        print_scores()
        nan_freq = t.sum(t.isnan(attn_probs)).item() / attn_probs.numel()
        raise ValueError(
            f"Your post-softmax masked attention scores contains {nan_freq * 100}% NaNs. Make sure you aren't setting an entire row to neg inf."
        )

    if not t.allclose(actual, expected):
        print_scores()
        t.testing.assert_close(actual, expected)

    print("All tests in `test_causal_mask` passed!")