<div align="center">

# Cliff Tokens: Identifying Single-Token Failure Triggers in LLM Mathematical Reasoning

📃 **Paper Link**: Coming soon

**Jaeyong Ko**¹, **Pilsung Kang**¹, **Yukyung Lee**²†

¹Seoul National University, ²Boston University

†Corresponding author

</div>

<p align="center">
  <img src="paper_images/main_figure.png" alt="Cliff Tokens main figure" width="900">
</p>

<div align="center">
<sub><i>Overview of Cliff Tokens. We estimate token-wise potential with rollouts and identify the precise token where a reasoning trace shifts toward failure under an adaptive one-sided z-test threshold.</i></sub>
</div>

## Abstract

Large language models (LLMs) reach high accuracy in mathematical reasoning, but individual traces on the same problem diverge; some arrive at the correct answer while others fail. Prior work analyzes failure at the step, chunk, or sentence level, or at tokens where failure has already occurred. Neither identifies the precise token that triggers the shift toward failure. We introduce the **cliff token**, a token where the token-wise potential drops significantly under an adaptive threshold that scales with the local token-wise potential, based on a one-sided two-proportion z-test. Across seven models and three mathematical reasoning benchmarks (GSM1K, MATH500, AIME 2025), cliff tokens act as failure triggers; deleting the first cliff token and resampling recovers pass@64 to 1.0, while keeping it limits recovery to 0.71–1.00. We further introduce a cliff taxonomy of deterministic, uncertain, and sampled-off cliffs, defined by greedy choice and token entropy. Each type has distinct probabilistic characteristics, and the taxonomy generalizes across model scales. Finally, we validate the taxonomy via single-token preference optimization at cliff positions (Cliff-DPO). Trained on GSM8K, Cliff-DPO improves accuracy across benchmarks by up to +6.6. Optimizing at uncertain and sampled-off cliffs improves reasoning, while deterministic cliffs do not.


## Paper in Brief

- **Token-Wise Potential.** The probability that a reasoning process reaches the correct answer, given the partial trace up to token position `t`.
- **Cliff Token.** A token whose rollout-estimated potential drops significantly under the adaptive threshold `Δ_t > 0.1 + 1.645 · SE_t`.
- **RQ1. Failure Trigger.** Cliff tokens occur more often in incorrect traces; deleting the first cliff token (`Cliff-del`) restores reasoning more reliably than continuing from it (`Cliff-keep`).
- **RQ2. Cliff Taxonomy.** Cliff tokens are categorized by greedy choice and token entropy into deterministic, uncertain, and sampled-off cliffs.
- **RQ3. Family and Scale Effects.** Deterministic cliffs are largely scale-invariant, uncertain cliffs expose model-specific knowledge gaps, and sampled-off cliffs show scale-asymmetry.
- **Cliff-DPO.** Single-token preference optimization at cliff positions improves reasoning when trained on uncertain and sampled-off cliffs, while deterministic cliffs do not.

<p align="center">
  <img src="paper_images/fig16_passk_incorrect.png" alt="Cliff-del vs. Cliff-keep representative result" width="900">
</p>

<div align="center">
<sub><i>RQ1. Failure Trigger. Cliff-del removes the first cliff token and resamples, while Cliff-keep continues from the fixed cliff token. The recovery gap shows that a single cliff token can trigger reasoning failure.</i></sub>
</div>

## Project Structure
### Repository Layout

```text
.
├── src/                 # core Python package
├── scripts/             # experiment entrypoints
├── figure/              # figure notebooks, reduced data, generated figures
├── paper_images/        # exact PDF images used in the paper
├── requirements.txt     # tested Python/CUDA dependency pins
├── README.md
└── LICENSE
```


### Output Layout

```text
output/
├── 01_inference/                # sampled reasoning traces
├── 02_token_stats/              # per-token logprob/rank/entropy stats
├── 03_rollout/                  # token-wise potential rollout outputs
├── 04_cliff_occurrence/         # cliff occurrence and taxonomy summaries
├── 05_deletion_ablation/        # Cliff-del / Cliff-keep pass@k results
├── 06_entropy_rank/             # entropy/rank analyses around cliffs
├── 07_candidate_replacement/    # candidate replacement at cliff positions
├── 08_cpm_shift/                # cross-model cliff probability mass shift
└── 09_cliff_dpo/
    ├── 01_candidates/           # top-k candidate rollout at cliff positions
    ├── 02_pairs/                # cliff-position preference pairs
    ├── 03_training/             # trained Cliff-DPO adapters
    ├── 04_eval/                 # adapter evaluation outputs
    ├── 05_cliff_count/          # post-training cliff-count evaluation
    └── logs/
```


## 🛠️ Installation

```bash
git clone https://github.com/beaver-22/Cliff-token.git
cd Cliff-token
```
### Env Setup
```bash
conda create -n cliff python=3.10 -y
conda activate cliff
pip install -r requirements.txt
```


## 🚀 Reproduction

### Prepare

```bash
export GPU_IDS=0
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export HF_TOKEN=hf_xxx  # for gated Llama/Gemma models

python -m src.utils.download_models --hf_token "$HF_TOKEN"
python -m src.utils.download_datasets --dataset gsm1k math500 aime25 gsm8k
python -m src.utils.create_subsets --seed 42
```

### Inference and Rollout

1. Generate sampled reasoning traces for the target model and datasets.

```bash
bash scripts/run_inference.sh \
  --model qwen3-0.6b \
  --dataset gsm1k_100,math500_100,aime25 \
  --gpus "$GPU_IDS" \
  --output_dir output/01_inference
```

2. Compute token-level logprob, rank, and entropy statistics
```bash
python3 scripts/_compute_token_stats.py \
  --gpu "$GPU_IDS" \
  --source output/01_inference \
  --output_dir output/02_token_stats \
  --skip-existing
```

3. Estimate token-wise potential by rollout sampling. The paper uses `N=64` rollouts per token position.

```bash
bash scripts/run_rollout.sh \
  --model qwen3-0.6b \
  --dataset gsm1k_100 \
  --data_path output/01_inference/Qwen3-0.6B/gsm1k_100_all_paths.json \
  --rollout_samples 64 \
  --gpus "$GPU_IDS" \
  --output_dir output/03_rollout/Qwen3-0.6B
```


### Analysis

RQ1 measures cliff occurrence and tests whether the first cliff token is a failure trigger.

```bash
bash scripts/run_exp1_occurrence.sh \
  --rollout_dir output/03_rollout \
  --datasets gsm1k_100,math500_100,aime25 \
  --output_dir output/04_cliff_occurrence/paper
```
```bash
bash scripts/run_exp1_deletion.sh \
  --rollout_dir output/03_rollout \
  --datasets gsm1k_100,math500_100,aime25 \
  --gpus "$GPU_IDS" \
  --output_dir output/05_deletion_ablation/paper_batch
```

RQ2 analyzes entropy/rank behavior and assigns deterministic, uncertain, or sampled-off cliff categories.

```bash
bash scripts/run_exp3_entropy.sh \
  --rollout_dir output/03_rollout \
  --baseline_dir output/02_token_stats \
  --datasets gsm1k_100,math500_100,aime25 \
  --gpus "$GPU_IDS" \
  --output_dir output/06_entropy_rank/paper_batch
```

RQ2/RQ3 evaluate candidate replacement and cross-model cliff probability mass shift.

```bash
bash scripts/run_exp4_candidates_all_models.sh \
  --gpus "$GPU_IDS" \
  --parallel_mode auto
```

```bash
bash scripts/run_exp5_cpm_shift.sh \
  --sources qwen3-0.6b,qwen3-8b \
  --evals qwen3-0.6b,qwen3-8b \
  --datasets gsm1k_100,math500_100,aime25 \
  --gpus "$GPU_IDS" \
  --output_dir output/08_cpm_shift/qwen_small_big_batch
```


## 🧗 Cliff-DPO

### 1. Candidate Rollout

```bash
bash scripts/run_dpo_rollout.sh \
  --model qwen3-0.6b \
  --dataset gsm8k \
  --data_path output/03_rollout/Qwen3-0.6B/gsm8k_all_paths.json \
  --gpus "$GPU_IDS" \
  --k_candidates 10 \
  --num_samples 64
```

### 2. Build Preference Pairs

```bash
python -m src.dpo.build_dpo_pairs \
  --candidates_path output/09_cliff_dpo/01_candidates/Qwen3-0.6B/gsm8k_cliff_candidates.json \
  --output_dir output/09_cliff_dpo/02_pairs/Qwen3-0.6B \
  --strategy cliff_1N \
  --category_ablations
```

### 3. Train

```bash
bash scripts/run_dpo_train.sh \
  --suite \
  --model ./model/Qwen3-0.6B \
  --dataset gsm8k \
  --gpus "$GPU_IDS" \
  --wandb_mode disabled
```

### 4. Evaluate

```bash
python -m src.dpo.evaluate \
  --model qwen3-0.6b \
  --adapter_paths none \
    output/09_cliff_dpo/03_training/Qwen3-0.6B/gsm8k/cliff_all \
    output/09_cliff_dpo/03_training/Qwen3-0.6B/gsm8k/cliff_deterministic_only \
    output/09_cliff_dpo/03_training/Qwen3-0.6B/gsm8k/cliff_uncertainty_only \
    output/09_cliff_dpo/03_training/Qwen3-0.6B/gsm8k/cliff_sampled_off_only \
    output/09_cliff_dpo/03_training/Qwen3-0.6B/gsm8k/cliff_uncertainty_sampled_off_only \
  --labels Baseline Cliff-all Cliff-deterministic Cliff-uncertainty Cliff-sampled-off Cliff-uncertainty-sampled-off \
  --full_suite \
  --token_profile paper \
  --aime_samples 64 \
  --gpus "$GPU_IDS" \
  --output_dir output/09_cliff_dpo/04_eval/Qwen3-0.6B
```


## 📄 License

The code in this repository is released under the MIT License; see `LICENSE`.

Downloaded model weights, datasets, and benchmark contents are governed by their original upstream licenses and terms of use. In particular, Llama and Gemma require accepting their HuggingFace license terms before download.


## 📚 Citation

```bibtex
@article{ko2026clifftoken,
  title={Cliff Tokens: Identifying Single-Token Failure Triggers in LLM Mathematical Reasoning},
  author={Ko, Jaeyong and Kang, Pilsung and Lee, Yukyung},
  journal={arXiv preprint},
  year={2026}
}
```
