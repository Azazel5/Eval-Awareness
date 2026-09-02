#!/usr/bin/env python3
"""Run Qwen3-32B (thinking enabled) over data/eval/triggers_augmented.json on a CUDA GPU.

For every row of the augmented triggers dataset, generates a completion for
both the `real` and `hypothetical` prompt under whatever `condition`
(eval/prod/control) that row already carries. Each generation is teacher-free
(no forced continuation) and captures the full `<think>...</think>` block
plus the final response separately.

Requires a CUDA host. Qwen3-32B in bf16 is ~65GB of weights — needs an
80GB GPU, or multiple GPUs with `--device-map auto`, or pass a quantized
checkpoint via `--model`.

Usage:

    python scripts/run_qwen3_inference.py \
        --dataset data/eval/triggers_augmented.json \
        --output data/results/qwen3_32b_triggers.jsonl \
        --batch-size 8 --max-new-tokens 1024

    # smoke test on a handful of rows before committing to a full run
    python scripts/run_qwen3_inference.py --limit 4 --batch-size 2

Resumable: if --output already exists, ids already present are skipped, so a
killed/interrupted run can just be re-launched with the same command.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from qwen3_utils import (
    DEFAULT_MODEL,
    THINKING_GENERATION_DEFAULTS,
    build_chat_prompt,
    load_qwen3,
    load_triggers_augmented,
    split_think_and_response,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "data" / "eval" / "triggers_augmented.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "results" / "qwen3_32b_triggers.jsonl"


def build_tasks(
    rows: list[dict[str, Any]], sides: list[str], conditions: list[str] | None
) -> list[dict[str, Any]]:
    tasks = []
    for row in rows:
        if conditions is not None and row["condition"] not in conditions:
            continue
        for side in sides:
            tasks.append(
                {
                    "id": f"{row['id']}-{side}",
                    "source_id": row["id"],
                    "condition": row["condition"],
                    "side": side,
                    "raw_prompt": row[side],
                    "entities": row[f"{side}_entities"],
                }
            )
    return tasks


def load_done_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done = set()
    with output_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--sides", default="real,hypothetical", help="comma-separated subset of {real,hypothetical}")
    parser.add_argument("--conditions", default=None, help="comma-separated subset of {eval,prod,control}; default all")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N tasks (smoke testing)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable-thinking", action="store_true", default=True)
    parser.add_argument("--no-thinking", dest="enable_thinking", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    sides = args.sides.split(",")
    conditions = args.conditions.split(",") if args.conditions else None

    rows = load_triggers_augmented(str(args.dataset))
    tasks = build_tasks(rows, sides, conditions)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    done_ids = load_done_ids(args.output)
    tasks = [t for t in tasks if t["id"] not in done_ids]
    print(f"{len(done_ids)} already done, {len(tasks)} remaining")
    if not tasks:
        return

    print(f"loading {args.model} (dtype={args.dtype}, attn={args.attn_implementation}) …")
    loaded = load_qwen3(
        model_name=args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    )
    model, tokenizer = loaded.model, loaded.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    t_start = time.time()

    with args.output.open("a", encoding="utf-8") as out_handle:
        for batch_start in range(0, len(tasks), args.batch_size):
            batch = tasks[batch_start : batch_start + args.batch_size]
            prompts = [
                build_chat_prompt(tokenizer, t["raw_prompt"], enable_thinking=args.enable_thinking)
                for t in batch
            ]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(loaded.device)
            input_len = inputs["input_ids"].shape[1]

            gen_kwargs = dict(THINKING_GENERATION_DEFAULTS) if args.enable_thinking else {"do_sample": False}
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    **gen_kwargs,
                )

            new_tokens = out[:, input_len:]
            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=False)

            for task, text in zip(batch, texts):
                reasoning, response = split_think_and_response(text)
                record = {
                    **task,
                    "model": args.model,
                    "enable_thinking": args.enable_thinking,
                    "reasoning": reasoning,
                    "response": response,
                    "raw_generated_text": text,
                }
                out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
            out_handle.flush()

            elapsed = time.time() - t_start
            rate = written / elapsed if elapsed > 0 else 0.0
            print(f"  {batch_start + len(batch)}/{len(tasks)} tasks  ({rate:.2f}/s)")

    print(f"wrote {written} generations to {args.output}")


if __name__ == "__main__":
    main()
