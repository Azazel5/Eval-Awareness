"""Shared Qwen3 loading/formatting helpers for run_qwen3_inference.py and atp_star.py.

Kept model-loading logic in one place so the inference run and the circuit
analysis run always use the same model config (dtype, attention
implementation, chat template).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

DEFAULT_MODEL = "Qwen/Qwen3-32B"

# Qwen3's own recommended sampling params for thinking mode (see the model
# card: greedy decoding is prone to repetition/degeneration for this model).
THINKING_GENERATION_DEFAULTS = {
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
}

THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


@dataclass
class LoadedModel:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device


def load_qwen3(
    model_name: str = DEFAULT_MODEL,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    attn_implementation: str = "sdpa",
    gradient_checkpointing: bool = False,
    load_in_8bit: bool = False,
) -> LoadedModel:
    """Load Qwen3 for CUDA inference.

    attn_implementation:
      - "sdpa" / "flash_attention_2": fast, no exposed attention weights.
        Use for run_qwen3_inference.py.
      - "eager": slower, but exposes per-head attention weights needed by
        atp_star.py's QK-fix. Use for the ATP* run.

    load_in_8bit: quantize weights via bitsandbytes (`pip install
    bitsandbytes` if not already present). Qwen3-32B is ~65GB in bf16 —
    on a single 80GB GPU that alone leaves only ~15GB for activations,
    the eager-attention QK-fix machinery, and backward, which isn't
    enough (confirmed by OOM even with gradient_checkpointing +
    attn_implementation="sdpa"). 8-bit roughly halves the weight
    footprint to ~32GB, which is the actual fix for that — THIS is the
    lever for an OOM on atp_star.py, not gradient_checkpointing.

    gradient_checkpointing: recompute activations during backward instead
    of storing all of them. DO NOT USE with atp_star.py currently — its
    Qwen3Capture hooks call .retain_grad() during the forward pass, then
    the hooks are removed before .backward() runs. Under checkpointing,
    the backward-relevant activations get recomputed from scratch inside
    .backward(), which our (already-removed) hooks never see — so
    clean_act.grad would silently come back None for every layer, and
    compute_atp_star()'s `if grad is None: continue` would silently skip
    scoring almost everything. That's wrong-results, not a crash, so this
    flag is left here but should stay off for atp_star.py until that hook
    lifetime is fixed to also cover .backward(). Safe for anything that
    only calls .backward() without depending on hook-captured
    intermediate activations.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this environment. This script is meant "
            "to run on a CUDA GPU host; a 32B model is not practical on CPU/MPS."
        )
    if gradient_checkpointing:
        import warnings

        warnings.warn(
            "gradient_checkpointing=True: see load_qwen3's docstring — this is "
            "known-broken with atp_star.py's hook-based gradient capture "
            "(silently zeroes out attn_out/mlp_out scores). Use load_in_8bit "
            "instead for an OOM on that script.",
            stacklevel=2,
        )

    torch_dtype = getattr(torch, dtype)
    quantization_config = None
    if load_in_8bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        quantization_config=quantization_config,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # incompatible with checkpointing
    model.eval()
    device = next(model.parameters()).device
    return LoadedModel(model=model, tokenizer=tokenizer, device=device)


def build_chat_prompt(
    tokenizer: PreTrainedTokenizerBase,
    user_text: str,
    enable_thinking: bool = True,
) -> str:
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def split_think_and_response(generated_text: str) -> tuple[str | None, str]:
    """Split a Qwen3 completion into (reasoning_trace, final_response).

    Qwen3 emits `<think>...</think>` before the final answer when thinking is
    enabled. If no think block is found (e.g. it was truncated by
    max_new_tokens, or thinking was off), reasoning is None and the whole
    text is treated as the response.
    """
    match = THINK_BLOCK_RE.search(generated_text)
    if match is None:
        return None, generated_text.strip()
    reasoning = match.group(1).strip()
    response = generated_text[match.end():].strip()
    return reasoning, response


def load_triggers_augmented(path: str) -> list[dict[str, Any]]:
    import json

    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
