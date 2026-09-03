# Eval-Awareness

Does Qwen3 internally represent "I might be being evaluated", and if so, can that representation be causally localized to specific components, rather than just read off its verbalized reasoning?

## Goal

Language models often (not always) mention when they suspect they're being tested, prior work has shown this happens inconsistently, and reading transcripts alone can't tell you whether a "miss" means the model genuinely didn't notice or just didn't say so. This project isolates two plausible triggers for eval-awareness, an explicit environment-metadata tag, and whether the request's entities look real or like placeholders, and asks whether swapping either one, with everything else held fixed, moves anything inside the model in a way that's causally verifiable, not just correlational.

## Part 1 — Data

Starting point is [`microsoft/Test_Awareness_Steering`](https://github.com/microsoft/Test_Awareness_Steering)'s `triggers.json`: 2,587 paired requests, each with a `real` version (named entities, real URLs) and a `hypothetical` version (dummy entities like "Jane Doe", `.test` domains). `data/augment_triggers.py` adds a second, independent axis on top: every row gets rewritten three times with a bracketed `[ENV_METADATA: ...]` tag prepended, `eval` (`MODE=EVAL_SAFETY`), `prod` (`MODE=INFERENCE_GENERAL`), or `control` (`MODE=STANDARD_CHAT`), giving `data/eval/triggers_augmented.json` at 7,761 rows (2,587 × 3 tags).

`scripts/run_qwen3_inference.py` runs Qwen3-32B in thinking mode over this dataset and saves each `<think>...</think>` trace plus final response. The canonical use is generating CoTs under the `eval` tag only (`--conditions eval`), those get teacher-forced onto the `control`/`prod` versions of the same row before circuit analysis, so a clean/corrupt pair differs *only* in the tag, never in what the model actually reasoned about.

## Part 2 — Circuit analysis

`scripts/atp_star.py` implements Attribution Patching (AtP*, [Kramár et al. 2024](https://arxiv.org/abs/2403.00745)) to localize which layers' activations shift when one axis is swapped, e.g. `control` vs `eval` tag, same entities, same forced CoT. For each pair it runs one gradient-tracked forward pass on the clean input and one plain forward pass on the corrupted input, and estimates each node's causal contribution as `(corrupt_activation − clean_activation) · gradient`, aggregated over thousands of pairs. The top-ranked nodes then get checked against ground truth: a real forward pass with that node's activation actually substituted, not just the linear estimate, so a spurious estimate is caught rather than trusted.

## Status

Exploratory and still being run against Qwen3-32B on a single 80GB GPU, the model needs 8-bit quantization to fit alongside AtP*'s backward pass, which currently means running without the paper's QK-fix correction (`--no-qk-fix`), so results so far cover layer-level `attn_out`/`mlp_out` localization, not per-head attention detail.

## Layout

```
data/
  eval/triggers.json              paired real/hypothetical requests (source)
  eval/triggers_augmented.json    + eval/prod/control tags (7,761 rows)
  augment_triggers.py             builds the file above
scripts/
  qwen3_utils.py                  shared model loading, chat template, CoT parsing
  run_qwen3_inference.py          batched Qwen3-32B generation over the dataset
  atp_star.py                     AtP* circuit localization + exact-patch verification
results/
  qwen3_32b_triggers.jsonl        generated CoTs (eval condition)
  atp_star_*.json                 AtP* rankings per axis
```
