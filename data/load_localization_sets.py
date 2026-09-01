#!/usr/bin/env python3
"""Load the three locked localization sets and print a head from each.

There was no loader file before this. Downloads were run ad hoc in a shell.

    A  BBQ              Hugging Face  walledai/BBQ
    B  FORTRESS         Hugging Face  ScaleAI/fortress_public
    F  Ferrando entities  GitHub only   javiferran/sae_entities

Usage (from the MATS Fall 2026 folder):

    python data/load_localization_sets.py
    python data/load_localization_sets.py --head 5 --budget 100 --seed 0
    python data/load_localization_sets.py --save

`--save` writes protocol samples to data/processed/*.jsonl (not the full dumps).
"""

from __future__ import annotations

import argparse
import json
import re
import ssl
import urllib.request
from pathlib import Path
from typing import Any

import certifi

from datasets import Dataset, concatenate_datasets, load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

BBQ_HF = "walledai/BBQ"
FORTRESS_HF = "ScaleAI/fortress_public"
FERRANDO_PROMPTS_URL = (
    "https://raw.githubusercontent.com/javiferran/sae_entities/main/"
    "dataset/processed/entity_prompts/gemma-2-2b-it_wikidata_splits.json"
)
FERRANDO_ENTITY_BASE = (
    "https://raw.githubusercontent.com/javiferran/sae_entities/main/"
    "dataset/processed/entities"
)
FERRANDO_ENTITY_TYPES = ("player", "movie", "city", "song")

# Explicit "this is an eval / jailbreak test" cues. Drop these from B so
# localization is not just reading the words "evaluated".
EXPLICIT_EVAL_RE = re.compile(
    r"("
    r"you are being evaluated"
    r"|you('re| are) (currently )?being (tested|evaluated)"
    r"|this is (a |an )?(test|eval|evaluation|benchmark)\b"
    r"|as part of (a |an )?(eval|evaluation|benchmark|red ?team)"
    r"|jailbreak"
    r"|red[- ]?team(ing)?"
    r"|ignore (all |previous |your )?instructions"
    r"|DAN mode"
    r")",
    re.IGNORECASE,
)

UNKNOWN_CHOICE_RE = re.compile(
    r"can't be determined|cannot be determined|not (enough|known)|unknown|"
    r"not enough info|undetermined|can't answer",
    re.IGNORECASE,
)


def _trunc(value: Any, n: int = 220) -> str:
    if isinstance(value, (list, dict)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value).replace("\n", " ")
    return text if len(text) <= n else text[:n] + "…"


def print_head(title: str, rows: list[dict[str, Any]], n: int = 5) -> None:
    print("\n" + "=" * 88)
    print(title)
    print(f"n={len(rows)}  showing {min(n, len(rows))}")
    print("=" * 88)
    for i, row in enumerate(rows[:n]):
        print(f"\n--- row {i} ---")
        for key, value in row.items():
            if key == "raw":
                continue
            print(f"  {key}: {_trunc(value)}")


def letter_choices(choices: list[str]) -> str:
    return "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))


def is_ambiguous_bbq(choices: list[str], answer: int) -> bool:
    if answer < 0 or answer >= len(choices):
        return False
    return bool(UNKNOWN_CHOICE_RE.search(choices[answer]))


def format_bbq_prompt(context: str, question: str, choices: list[str]) -> str:
    return (
        f"{context.strip()}\n\n"
        f"{question.strip()}\n"
        f"{letter_choices(choices)}"
    )


def load_bbq() -> Dataset:
    """All 11 BBQ category splits, concatenated. ~58k MCQ items."""
    bundle = load_dataset(BBQ_HF)
    parts = [bundle[split] for split in bundle]
    return concatenate_datasets(parts)


def bbq_records(ds: Dataset) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, row in enumerate(ds):
        choices = list(row["choices"])
        answer = int(row["answer"])
        out.append(
            {
                "id": f"bbq-{row['category']}-{i}",
                "role": "A",
                "source": BBQ_HF,
                "category": row["category"],
                "ambiguous": is_ambiguous_bbq(choices, answer),
                "context": row["context"],
                "question": row["question"],
                "choices": choices,
                "answer_index": answer,
                "gold": choices[answer] if 0 <= answer < len(choices) else None,
                "prompt": format_bbq_prompt(row["context"], row["question"], choices),
            }
        )
    return out


def load_fortress() -> Dataset:
    return load_dataset(FORTRESS_HF, split="train")


def fortress_looks_like_explicit_eval(text: str) -> bool:
    return bool(EXPLICIT_EVAL_RE.search(text or ""))


def fortress_records(ds: Dataset) -> list[dict[str, Any]]:
    """Eval B uses the adversarial prompt (this *is* a safety eval).

    Drop rows whose text already says it is an eval/jailbreak, so the
    circuit search is not a lexical detector for the word 'evaluated'.
    """
    out: list[dict[str, Any]] = []
    for row in ds:
        prompt = row["adversarial_prompt"] or ""
        dropped = fortress_looks_like_explicit_eval(prompt) or (
            fortress_looks_like_explicit_eval(row.get("benign_prompt") or "")
        )
        out.append(
            {
                "id": f"fortress-{row['ID']}",
                "role": "B",
                "source": FORTRESS_HF,
                "fortress_id": row["ID"],
                "risk_domain": row["risk_domain"],
                "risk_subdomain": row["risk_subdomain"],
                "prompt": prompt,
                "benign_prompt": row["benign_prompt"],
                "rubric_n": len(row["rubric"]) if row.get("rubric") else 0,
                "dropped_explicit_eval": dropped,
            }
        )
    return out


def _http_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "localization-sets/1.0"})
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(req, timeout=120, context=context) as response:
        return json.loads(response.read().decode("utf-8"))


def load_ferrando_prompts() -> dict[str, list[str]]:
    """Open-ended attribute questions. Gemma known/unknown labels do not transfer."""
    raw = _http_json(FERRANDO_PROMPTS_URL)
    if not isinstance(raw, dict):
        raise TypeError(f"unexpected Ferrando splits JSON: {type(raw)}")
    return {split: list(questions) for split, questions in raw.items()}


def load_ferrando_entities() -> dict[str, list[dict[str, Any]]]:
    catalogs: dict[str, list[dict[str, Any]]] = {}
    for entity_type in FERRANDO_ENTITY_TYPES:
        catalogs[entity_type] = _http_json(f"{FERRANDO_ENTITY_BASE}/{entity_type}.json")
    return catalogs


def ferrando_records(splits: dict[str, list[str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split, questions in splits.items():
        for i, question in enumerate(questions):
            out.append(
                {
                    "id": f"ferrando-{split}-{i}",
                    "role": "F",
                    "source": "javiferran/sae_entities",
                    "split": split,
                    "prompt": question,
                    "wrapper": "neutral",
                }
            )
    return out


def sample_records(
    rows: list[dict[str, Any]], budget: int, seed: int
) -> list[dict[str, Any]]:
    if len(rows) <= budget:
        return list(rows)
    return Dataset.from_list(rows).shuffle(seed=seed).select(range(budget)).to_list()


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", type=int, default=5, help="rows to print per set")
    parser.add_argument(
        "--budget",
        type=int,
        default=100,
        help="protocol sample size per set (~50–100 in EXPERIMENT.md)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save",
        action="store_true",
        help=f"write samples under {PROCESSED_DIR}",
    )
    parser.add_argument(
        "--skip-entity-catalog",
        action="store_true",
        help="do not download Ferrando player/movie/city/song JSON (large)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"A  loading {BBQ_HF} …")
    bbq_ds = load_bbq()
    bbq = bbq_records(bbq_ds)
    print(f"   {len(bbq)} items across categories")

    print(f"B  loading {FORTRESS_HF} …")
    fortress_ds = load_fortress()
    fortress_all = fortress_records(fortress_ds)
    fortress = [r for r in fortress_all if not r["dropped_explicit_eval"]]
    print(
        f"   {len(fortress_all)} public items, "
        f"{len(fortress)} after dropping explicit-eval wording "
        f"({len(fortress_all) - len(fortress)} dropped)"
    )

    print("F  loading Ferrando prompts from GitHub (not on Hugging Face) …")
    splits = load_ferrando_prompts()
    ferrando = ferrando_records(splits)
    print(
        "   prompt splits "
        + ", ".join(f"{k}={len(v)}" for k, v in splits.items())
        + f"  total={len(ferrando)}"
    )

    catalogs: dict[str, list[dict[str, Any]]] = {}
    if not args.skip_entity_catalog:
        print("F  loading Ferrando entity catalogs (player/movie/city/song) …")
        catalogs = load_ferrando_entities()
        print(
            "   entities "
            + ", ".join(f"{k}={len(v)}" for k, v in catalogs.items())
        )

    print_head(f"A  BBQ  {BBQ_HF}", bbq, args.head)
    print_head(
        f"B  FORTRESS  {FORTRESS_HF}  (explicit-eval rows dropped; prompts truncated in print)",
        fortress,
        args.head,
    )
    print_head("F  Ferrando prompts  GitHub gemma-2-2b-it_wikidata_splits.json", ferrando, args.head)
    if catalogs:
        player_head = [
            {
                "entity": row["entity"],
                "entity_type": row["entity_type"],
                "attributes": row["attributes"],
            }
            for row in catalogs["player"][: args.head]
        ]
        print_head("F  Ferrando entity catalog  player.json", player_head, args.head)

    protocol = {
        "A_bbq": sample_records(bbq, args.budget, args.seed),
        "B_fortress": sample_records(fortress, args.budget, args.seed),
        "F_ferrando": sample_records(ferrando, args.budget, args.seed),
    }
    print("\n" + "=" * 88)
    print(
        f"protocol samples  budget={args.budget}  seed={args.seed}  "
        + "  ".join(f"{k}={len(v)}" for k, v in protocol.items())
    )
    print("Hour 0: run these on Qwen3-14B thinking. Kill if A or B verbalize ~0, or F does not.")
    print("=" * 88)

    if args.save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        for name, rows in protocol.items():
            path = PROCESSED_DIR / f"{name}.jsonl"
            save_jsonl(path, rows)
            print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()