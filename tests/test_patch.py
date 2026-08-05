"""Tests for component capture and patching.

The load-bearing one is `test_patching_a_component_with_its_own_value_is_identity`.
Every effect this project reports is a difference between a clean run and a patched
run, so a patcher that perturbs anything beyond the requested slice inflates every
number downstream while still looking like it works.

A tiny real architecture is instantiated from config with random weights. That
exercises the actual module paths (`self_attn.o_proj`, `mlp`) rather than a stand-in
that happens to have the right attribute names.
"""

from __future__ import annotations

import pytest
import torch

from innerj.patch import Component, capture, head_width, run_patched


@pytest.fixture(scope="module")
def tiny_model():
    """A 4-layer Llama with grouped-query attention, on CPU with random weights.

    GQA is deliberate: with `num_key_value_heads < num_attention_heads` the
    head-slicing arithmetic has a chance to be wrong in a way that a
    single-key-value-head model would hide.
    """
    from jlens import from_hf
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        # Matched to the GPT-2 tokenizer's id range; everything else stays tiny.
        vocab_size=50257,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    hf_model = LlamaForCausalLM(config).eval()
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    return from_hf(hf_model, tokenizer, force_bos=False)


PROMPT = "the quick brown fox jumps over the lazy dog and keeps running onward"
OTHER = "meanwhile a silent grey heron waited beside the cold river all morning"


def test_component_validates_the_head_index():
    with pytest.raises(ValueError, match="take no head index"):
        Component("mlp", 3, head=0)
    assert str(Component("attn", 3, head=7)) == "attn.L3.H7"
    assert str(Component("attn", 3)) == "attn.L3.all"
    assert str(Component("mlp", 3)) == "mlp.L3"
    assert Component("attn", 3, head=7).slices_a_head
    assert not Component("attn", 3).slices_a_head


def test_head_width_comes_from_the_projection_not_the_config(tiny_model):
    """`n_heads * head_dim` need not equal `d_model` under GQA.

    Deriving the slice width from `head_dim` would misalign every head patch on a
    checkpoint where `o_proj` is not square.
    """
    width = head_width(tiny_model, 0)
    o_proj = tiny_model.layers[0].self_attn.o_proj
    assert width * 8 == o_proj.in_features


@pytest.mark.parametrize(
    "component",
    [
        Component("resid", 1),
        Component("mlp", 2),
        Component("attn", 2, head=5),
        Component("attn", 2),
    ],
)
def test_patching_a_component_with_its_own_value_is_identity(tiny_model, component):
    """The check that makes every downstream difference trustworthy.

    Capture from a prompt, patch back into the same prompt: the logits must be
    unchanged. A patcher that writes a misaligned slice, drops a tuple element, or
    mutates a shared tensor fails here and nowhere else.
    """
    clean, _ = run_patched(tiny_model, PROMPT, {})
    captured = capture(tiny_model, PROMPT, [component])
    patched, _ = run_patched(tiny_model, PROMPT, captured)
    assert torch.allclose(clean, patched, atol=1e-5)


@pytest.mark.parametrize(
    "component",
    [
        Component("resid", 1),
        Component("mlp", 2),
        Component("attn", 2, head=5),
        Component("attn", 2),
    ],
)
def test_patching_from_another_prompt_changes_the_output(tiny_model, component):
    """A patch that changes nothing would make every null result meaningless."""
    clean, _ = run_patched(tiny_model, PROMPT, {})
    donor = capture(tiny_model, OTHER, [component])
    patched, _ = run_patched(tiny_model, PROMPT, donor)
    assert not torch.allclose(clean, patched, atol=1e-4)


def test_hooks_do_not_leak_into_later_runs(tiny_model):
    """A surviving hook silently corrupts every subsequent measurement."""
    before, _ = run_patched(tiny_model, PROMPT, {})
    donor = capture(tiny_model, OTHER, [Component("resid", 1)])
    run_patched(tiny_model, PROMPT, donor)
    after, _ = run_patched(tiny_model, PROMPT, {})
    assert torch.allclose(before, after, atol=1e-6)


def test_capture_and_patch_agree_on_shape(tiny_model):
    captured = capture(
        tiny_model, PROMPT, [Component("attn", 1, head=3), Component("mlp", 1)]
    )
    width = head_width(tiny_model, 1)
    assert captured[Component("attn", 1, head=3)].shape == (1, width)
    assert captured[Component("mlp", 1)].shape == (1, tiny_model.d_model)


def test_multiple_positions_are_captured_and_restored(tiny_model):
    component = Component("mlp", 2)
    positions = [-3, -2, -1]
    clean, _ = run_patched(tiny_model, PROMPT, {}, positions=positions)
    captured = capture(tiny_model, PROMPT, [component], positions=positions)
    assert captured[component].shape[0] == 3
    patched, _ = run_patched(tiny_model, PROMPT, captured, positions=positions)
    assert torch.allclose(clean, patched, atol=1e-5)


def test_out_of_range_position_raises(tiny_model):
    with pytest.raises(ValueError, match="out of range"):
        capture(tiny_model, PROMPT, [Component("mlp", 0)], positions=[10_000])


def test_recorded_layers_come_back(tiny_model):
    _, recorded = run_patched(tiny_model, PROMPT, {}, record_layers=[0, 2])
    assert set(recorded) >= {0, 2, tiny_model.n_layers - 1}
    assert recorded[0].shape == (1, tiny_model.d_model)
