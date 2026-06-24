"""
Step 3: Cliff-DPO Training with TRL + LoRA

Supports:
  - Cliff-DPO token-level (cliff_1N, cliff_hard; deterministic/uncertain/sampled_off ablations)
  - All hyperparameters configurable via CLI

Usage:
    python -m src.dpo.train_dpo \
        --model ./model/Qwen3-0.6B \
        --dataset_path ./output/09_cliff_dpo/02_pairs/Qwen3-0.6B/cliff_all_gsm8k.json \
        --output_dir ./output/09_cliff_dpo/03_training/Qwen3-0.6B/gsm8k/cliff_all \
        --beta 0.1 --lr 5e-7 --lora_r 16

    # Evaluation is NOT run here. After training, run:
    #   python -m src.dpo.evaluate --model ./model/Qwen3-0.6B \
    #       --adapter_paths <trained_output_dir> --datasets gsm8k [gsm1k math500 aime25]
"""

import argparse
import json
import logging
import os
import sys
from typing import Optional, Dict

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer as _HFTrainer
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig

from src import config
# trl 0.9.6 × transformers 4.46+ name clash: trl's DPOTrainer defines
# get_batch_samples(self, model, batch) for its `generate_during_eval`
# helper, but transformers 4.46 added an unrelated
# get_batch_samples(self, epoch_iterator, num_batches, device) which the
# training loop calls every step. The trl override shadows it and the
# signatures don't match → TypeError on step 0. We don't use
# generate_during_eval, so restore the parent (transformers) implementation.
DPOTrainer.get_batch_samples = _HFTrainer.get_batch_samples

from src.dpo.logging_utils import parse_log_level, setup_logger

logger = logging.getLogger("dpo.step4_train")


# Silence trl 0.9.6 × transformers 4.46+ noise: trl's DPOTrainer accesses
# self.tokenizer in a tight per-row loop inside dataset.map(tokenize_row),
# and transformers turned that attribute into a per-access logger.warning.
# Drop just that one message from the transformers.trainer logger so real
# warnings still come through.
class _DropDeprecatedTokenizerWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Trainer.tokenizer is now deprecated" not in record.getMessage()


logging.getLogger("transformers.trainer").addFilter(_DropDeprecatedTokenizerWarning())


# ============================================================
# Token-level DPO trainer: bypasses BPE re-tokenization
# ============================================================

class _Trl096CompatMixin:
    """Shared shim for trl 0.9.6 × transformers 4.46+ signature drift.

    transformers 4.46+ added extra positional/keyword arguments to several
    Trainer hooks that trl 0.9.6's DPOTrainer overrides with the older
    signatures. We absorb the new arguments here and forward to trl's
    implementation so the metric computation/logging stays exactly as trl
    intends.

    Drift covered so far:
      - compute_loss(..., num_items_in_batch=...)  [transformers 4.46]
      - log(logs, start_time)                      [transformers 4.46]
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return super().compute_loss(model, inputs, return_outputs=return_outputs)

    def log(self, logs, start_time=None):
        # trl 0.9.6's DPOTrainer.log only takes `logs`; transformers 4.46+
        # passes a wall-clock `start_time` through. Drop it before delegating.
        return super().log(logs)


class TokenLevelDPOTrainer(_Trl096CompatMixin, DPOTrainer):
    """DPOTrainer subclass that uses pre-stored token IDs.

    Standard DPOTrainer.tokenize_row tokenizes (prompt + chosen) and
    (prompt + rejected) as strings, which can mis-align at the BPE
    boundary — turning a single "chosen" token into 0 or 2+ tokens.

    This override uses prompt_token_ids / chosen_token_id / rejected_token_id
    fields directly, guaranteeing that the chosen/rejected portion is exactly
    ONE token. The DPO loss therefore flows on exactly the cliff token.

    Prompt tokens are still masked via label_pad_token_id (standard DPO).
    """

    def tokenize_row(self, feature, model=None) -> Dict:
        # Fallback to standard tokenization if pre-tokenized fields missing
        if "prompt_token_ids" not in feature or "chosen_token_id" not in feature:
            return super().tokenize_row(feature, model)

        prompt_ids = list(feature["prompt_token_ids"])
        chosen_id = int(feature["chosen_token_id"])
        rejected_id = int(feature["rejected_token_id"])

        # transformers >= 4.46 turned Trainer.tokenizer into a deprecated
        # property that warns on every access; prefer processing_class.
        tok = getattr(self, "processing_class", None) or self.tokenizer
        bos = tok.bos_token_id
        eos = tok.eos_token_id
        pad_label = self.label_pad_token_id

        # Add BOS if missing
        if bos is not None and (len(prompt_ids) == 0 or prompt_ids[0] != bos):
            prompt_ids = [bos] + prompt_ids

        # Truncate prompt if too long (keep end - more relevant context)
        if len(prompt_ids) + 2 > self.max_length:
            keep = self.max_prompt_length
            if self.truncation_mode == "keep_start":
                prompt_ids = prompt_ids[:keep]
            else:
                prompt_ids = prompt_ids[-keep:]
            # Re-add BOS if truncation removed it
            if bos is not None and prompt_ids[0] != bos:
                prompt_ids = [bos] + prompt_ids[1:]

        # Build chosen sequence: prompt + [chosen] + [eos]
        # Loss flows ONLY on [chosen] + [eos]; prompt is masked.
        chosen_response_ids = [chosen_id]
        rejected_response_ids = [rejected_id]
        if eos is not None:
            chosen_response_ids.append(eos)
            rejected_response_ids.append(eos)

        chosen_input_ids = prompt_ids + chosen_response_ids
        rejected_input_ids = prompt_ids + rejected_response_ids

        chosen_attention_mask = [1] * len(chosen_input_ids)
        rejected_attention_mask = [1] * len(rejected_input_ids)

        # Labels: mask prompt portion
        chosen_labels = [pad_label] * len(prompt_ids) + chosen_response_ids
        rejected_labels = [pad_label] * len(prompt_ids) + rejected_response_ids

        return {
            "prompt_input_ids": prompt_ids,
            "prompt_attention_mask": [1] * len(prompt_ids),
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
            "rejected_labels": rejected_labels,
        }



# ============================================================
# Data loading
# ============================================================

def load_dpo_dataset(dataset_path: str, eval_split: float = 0.1):
    """Load DPO pairs JSON and split into train/eval.

    Auto-detects token-level mode: if pairs contain `prompt_token_ids`,
    `chosen_token_id`, `rejected_token_id`, those columns are also included
    so that TokenLevelDPOTrainer can bypass BPE re-tokenization.

    Returns: (train_ds, eval_ds, is_token_level)
    """
    with open(dataset_path) as f:
        pairs = json.load(f)

    logger.info(f"Loaded {len(pairs)} DPO pairs from {dataset_path}")

    is_token_level = (
        len(pairs) > 0
        and "prompt_token_ids" in pairs[0]
        and "chosen_token_id" in pairs[0]
        and "rejected_token_id" in pairs[0]
    )

    columns = {
        "prompt": [p["prompt"] for p in pairs],
        "chosen": [p["chosen"] for p in pairs],
        "rejected": [p["rejected"] for p in pairs],
    }
    if is_token_level:
        columns["prompt_token_ids"] = [p["prompt_token_ids"] for p in pairs]
        columns["chosen_token_id"] = [p["chosen_token_id"] for p in pairs]
        columns["rejected_token_id"] = [p["rejected_token_id"] for p in pairs]
        logger.info("  Mode: TOKEN-LEVEL (loss flows on exactly 1 token; prompt masked)")
    else:
        logger.info("  Mode: TRAJECTORY-LEVEL (no pre-tokenized fields found)")

    ds = Dataset.from_dict(columns)

    if eval_split > 0 and len(ds) > 10:
        split = ds.train_test_split(test_size=eval_split, seed=42)
        logger.info(f"  Train: {len(split['train'])}, Eval: {len(split['test'])}")
        return split["train"], split["test"], is_token_level
    else:
        logger.info(f"  Train: {len(ds)}, Eval: none")
        return ds, None, is_token_level


# ============================================================
# Training
# ============================================================

def train_dpo(
    model_path: str,
    dataset_path: str,
    output_dir: str,
    # DPO hyperparameters
    beta: float = 0.1,
    loss_type: str = "sigmoid",
    label_smoothing: float = 0.0,
    # Training hyperparameters
    learning_rate: float = 1e-6,
    lr_scheduler_type: str = "cosine",
    warmup_ratio: float = 0.1,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 16,
    max_length: int = 2048,
    max_prompt_length: int = 1024,
    max_grad_norm: float = 1.0,
    weight_decay: float = 0.0,
    # LoRA
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    # Eval
    eval_split: float = 0.1,
    eval_steps: int = 50,
    logging_steps: int = 10,
    save_steps: int = 100,
    # Misc
    seed: int = 42,
    bf16: bool = True,
    # wandb
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_tags: Optional[str] = None,
    wandb_mode: str = "online",
) -> str:
    """Run DPO training with LoRA."""
    os.makedirs(output_dir, exist_ok=True)

    # Save config for reproducibility
    train_config = {k: v for k, v in locals().items() if k != "self"}
    with open(os.path.join(output_dir, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=2)

    # ----- wandb setup -----
    use_wandb = wandb_project is not None
    if use_wandb:
        os.environ["WANDB_MODE"] = wandb_mode  # online / offline / disabled
        os.environ["WANDB_PROJECT"] = wandb_project
        if wandb_entity:
            os.environ["WANDB_ENTITY"] = wandb_entity
        if wandb_run_name is None:
            wandb_run_name = os.path.basename(output_dir.rstrip("/"))
        os.environ["WANDB_RUN_NAME"] = wandb_run_name
        if wandb_tags:
            os.environ["WANDB_TAGS"] = wandb_tags
        try:
            import wandb
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                entity=wandb_entity,
                tags=wandb_tags.split(",") if wandb_tags else None,
                config=train_config,
                dir=output_dir,
                reinit=True,
            )
            logger.info(f"[wandb] project={wandb_project} run={wandb_run_name} mode={wandb_mode}")
        except ImportError:
            logger.warning("[wandb] wandb not installed; training will run without logging")
            use_wandb = False
        except Exception as e:
            logger.warning(f"[wandb] init failed ({e}); training will run without logging")
            use_wandb = False

    logger.info("=" * 60)
    logger.info("DPO Training")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Dataset: {dataset_path}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Beta: {beta}, Loss: {loss_type}, LR: {learning_rate}")
    logger.info(f"  LoRA: r={lora_r}, alpha={lora_alpha}")
    logger.info("=" * 60)

    # Load dataset (auto-detect token-level vs trajectory-level)
    train_dataset, eval_dataset, is_token_level = load_dpo_dataset(dataset_path, eval_split)

    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # Apply LoRA
    target_modules = [m.strip() for m in lora_target_modules.split(",")]
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    logger.info(f"Applying LoRA: r={lora_r}, alpha={lora_alpha}, targets={target_modules}")

    # DPO training config
    training_args = DPOConfig(
        output_dir=output_dir,
        beta=beta,
        loss_type=loss_type,
        label_smoothing=label_smoothing,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        max_grad_norm=max_grad_norm,
        weight_decay=weight_decay,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=eval_steps if eval_dataset else None,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        seed=seed,
        bf16=bf16,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="wandb" if use_wandb else "none",
        run_name=wandb_run_name if use_wandb else None,
    )

    # TokenLevelDPOTrainer bypasses BPE re-tokenization so loss flows on
    # exactly one token (the cliff token). All Cliff-DPO pairs include
    # prompt_token_ids/chosen_token_id/rejected_token_id; the fallback path
    # inside tokenize_row handles any edge case where these fields are absent.
    trainer_cls = TokenLevelDPOTrainer
    logger.info(f"Using trainer: {trainer_cls.__name__}")
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
    )

    # Train
    logger.info("Starting DPO training...")
    trainer.train()

    # Save
    logger.info("Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Finish wandb run (auto-eval may start a fresh run)
    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    # Free CUDA memory before any in-process auto-eval (vLLM engine).
    # Without this, the trainer's model + optimizer + grads + KV state stay
    # resident on the GPU and vLLM OOMs while allocating its block table.
    import gc
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    logger.info(f"Training complete. Model saved to {output_dir}")
    return output_dir


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Cliff-DPO Training with TRL + LoRA")

    # Required
    parser.add_argument("--model", required=True, help="Base model path")
    parser.add_argument("--dataset_path", required=True, help="Path to DPO pairs JSON")
    parser.add_argument("--output_dir", required=True, help="Output directory")

    # DPO hyperparameters
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--loss_type", default="sigmoid",
                        choices=["sigmoid", "hinge", "ipo", "kto_pair"])
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", default="cosine",
                        choices=["cosine", "linear", "constant"])
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument(
        "--max_prompt_length", type=int, default=1024,
        help="Cap on prompt-side tokens. gsm8k prompts are p99~200 "
             "(text prompt) and ~733 (token-level cliff prefixes); 1024 covers "
             "both with headroom and frees ~512 extra tokens for the "
             "response budget vs the old default of 1536.",
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_targets",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated linear layer names to attach LoRA to. Default "
             "covers all attention + MLP projections (Qwen3 naming). "
             "Override to 'q_proj,v_proj' for the original LoRA paper's "
             "minimal config.",
    )

    # Eval
    parser.add_argument("--eval_split", type=float, default=0.1)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_bf16", action="store_true")

    # wandb
    parser.add_argument("--wandb_project", default=None,
                        help="wandb project name (e.g. 'cliff-dpo'). If unset, wandb disabled.")
    parser.add_argument("--wandb_run_name", default=None,
                        help="wandb run name. Default: output_dir basename")
    parser.add_argument("--wandb_entity", default=None, help="wandb team/user")
    parser.add_argument("--wandb_tags", default=None,
                        help="Comma-separated tags (e.g. 'qwen3-0.6b,strategy1,ss_only')")
    parser.add_argument("--wandb_mode", default="online",
                        choices=["online", "offline", "disabled"])

    # Logging
    parser.add_argument("--log_dir", default="./output/09_cliff_dpo/logs")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    # Setup logger
    global logger
    run_label = os.path.basename(args.output_dir.rstrip("/")) or "dpo_run"
    logger = setup_logger(
        name=f"step4_train_{run_label}",
        log_dir=args.log_dir,
        level=parse_log_level(args.log_level),
    )

    output_dir = train_dpo(
        model_path=args.model,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        beta=args.beta,
        loss_type=args.loss_type,
        label_smoothing=args.label_smoothing,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_targets,
        eval_split=args.eval_split,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        bf16=not args.no_bf16,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_entity=args.wandb_entity,
        wandb_tags=args.wandb_tags,
        wandb_mode=args.wandb_mode,
    )

    # NOTE: auto-eval has been removed. Running evaluate.py inside the same
    # GPU as a just-finished training run caused contention with other
    # concurrent training/auto-eval instances during grid sweeps, producing
    # corrupted accuracy numbers (see the v4 vs v5 cliff_1N_all eval where
    # the same adapter scored 68% in a clean eval and 54% during contention).
    # Run evaluation as a separate `python -m src.dpo.evaluate` command after
    # all training is done.


if __name__ == "__main__":
    main()
