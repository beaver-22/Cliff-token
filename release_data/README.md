# Stem Traces and Token-Level Rollouts

Reasoning traces and token-level success probabilities released with our paper. Each rollout record includes its stem trace, so stem traces are not stored twice. Run `python scripts/unpack_release_data.py` from the repository root to reconstruct both pipeline directories.

## How to use

```python
import gzip, json

with gzip.open("release_data/rollouts/Qwen3-8B/math500.json.gz", "rt", encoding="utf-8") as f:
    traces = json.load(f)

t = traces[0]
# t["all_position_scores"][i] = P(correct | response_token_ids[:i+1])
```

Files are gzip-compressed (lossless) JSON lists.

## Contents

```
rollouts/<model>/<dataset>.json.gz      # one sampled response per problem + token-level rollout scores
```

- **Models:** Qwen3-0.6B, Qwen3-4B, Qwen3-8B, Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct, gemma-3-4b-it
- **Datasets:** `gsm1k` (100), `math500` (100), and `aime25` (30) for all models; `gsm8k_train` (7,473) for Qwen3-0.6B, Llama-3.2-1B-Instruct, and Llama-3.1-8B-Instruct

The `all_position_scores` field can be omitted to recover the original stem-trace records.

## Fields

| Field | Description |
|---|---|
| `id`, `problem_id` | Trace and problem identifiers |
| `question`, `golden_answer` | Problem statement and list of accepted answers |
| `full_prompt` | Exact model input after the chat template |
| `response` | Generated stem trace |
| `response_tokens`, `response_token_ids` | Response tokens (strings / ids), length `n` |
| `total_tokens` | `n` |
| `is_correct` | Whether the stem trace's final answer is correct |
| `all_position_scores` | Length `n-1`; entry `i` is the fraction of 64 sampled continuations from prefix `response_token_ids[:i+1]` that reach the correct answer |

## Stem-trace accuracy

| Model | GSM1K | MATH500 | AIME25 | GSM8K train |
|---|---|---|---|---|
| Qwen3-0.6B | 49/100 | 48/100 | 2/30 | 5,022/7,473 |
| Qwen3-4B | 88/100 | 83/100 | 3/30 | – |
| Qwen3-8B | 88/100 | 85/100 | 7/30 | – |
| Llama-3.2-1B-Instruct | 30/100 | 23/100 | 0/30 | 3,686/7,473 |
| Llama-3.2-3B-Instruct | 65/100 | 45/100 | 0/30 | – |
| Llama-3.1-8B-Instruct | 71/100 | 44/100 | 1/30 | 6,632/7,473 |
| gemma-3-4b-it | 83/100 | 72/100 | 3/30 | – |
