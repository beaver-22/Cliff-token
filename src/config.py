"""
Configuration for Cliff Token Analysis (CTA)

This module contains:
- Few-shot prompts for GSM8K and MATH datasets
- Sampling configurations for thinking/non-thinking modes
- Model and dataset settings for Cliff Token Analysis
"""

from dataclasses import dataclass
from typing import Optional


# =============================================================================
# Few-shot Prompts
# =============================================================================

GSM8K_PROMPT = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Answer: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Answer: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.

"""


# Non-CoT: Direct answer only (no reasoning steps)
DIRECT_SYSTEM_PROMPT = "Follow the pattern and provide only the final answer. Do not show any work or reasoning."

GSM8K_DIRECT_PROMPT = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: 6

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: 39

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: 9

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: 29

"""

MATH_DIRECT_PROMPT = """Q: What is $15 \\times 12$?
A: $180$

Q: Solve for $x$: $2x + 10 = 20$.
A: $5$

Q: Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.
A: $[2,5)$

Q: If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$
A: $24$

"""


MATH_PROMPT = """Problem: Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}
Solution: The expressions inside each square root must be non-negative.
Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$.
Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$.
Therefore, the domain of the expression is $\\boxed{[2,5)}$.
Final Answer: The final answer is $[2,5)$. I hope it is correct.

Problem: If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$
Solution: We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$
Final Answer: The final answer is $24$. I hope it is correct.

Problem: Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?
Solution: If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$: \\begin{align*}
30n&=480\\\\
\\Rightarrow\\qquad n&=480/30=\\boxed{16}
\\end{align*}
Final Answer: The final answer is $16$. I hope it is correct.

Problem: If the system of equations

\\begin{align*}
6x-4y&=a,\\\\
6y-9x &=b.
\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ is nonzero.
Solution: If we multiply the first equation by $-\\frac{3}{2}$, we obtain

$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have

$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$
Final Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.

"""


# =============================================================================
# Zero-shot Prompts
# =============================================================================

# Qwen3 official style: instruction in system prompt, bare question in user message
ZEROSHOT_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

# =============================================================================
# Sampling Configuration
# =============================================================================

@dataclass
class SamplingConfig:
    """Sampling configuration for a specific mode."""
    temperature: float
    top_p: float
    top_k: int
    presence_penalty: float
    repetition_penalty: float
    max_tokens: int
    enable_thinking: bool
    use_chat_template: bool = True
    prompt_type: str = "zeroshot"  # "zeroshot" (default), "fewshot", or "direct"


# =============================================================================
# Model-specific Sampling Configurations
# =============================================================================

# Primary models for Cliff Token Analysis
# Keys are local paths (downloaded via download_models.py → ./model/<name>/)
MODEL_CONFIGS = {
    # --- Qwen3 models (support both thinking / non_thinking) ---
    "./model/Qwen3-0.6B": {
        "name": "Qwen3-0.6B",
        "thinking": SamplingConfig(
            temperature=0.6, top_p=0.95, top_k=20,
            presence_penalty=1.5, repetition_penalty=1.0,
            max_tokens=32768, enable_thinking=True,
        ),
        "non_thinking": SamplingConfig(
            temperature=0.7, top_p=0.8, top_k=20,
            presence_penalty=0.0, repetition_penalty=1.0,
            max_tokens=8192, enable_thinking=False,
        ),
    },
    "./model/Qwen3-4B": {
        "name": "Qwen3-4B",
        "thinking": SamplingConfig(
            temperature=0.6, top_p=0.95, top_k=20,
            presence_penalty=1.5, repetition_penalty=1.0,
            max_tokens=32768, enable_thinking=True,
        ),
        "non_thinking": SamplingConfig(
            # Official Qwen3 non-thinking defaults (technical report + HF model card)
            temperature=0.7, top_p=0.8, top_k=20,
            presence_penalty=0.0, repetition_penalty=1.0,
            max_tokens=8192, enable_thinking=False,
        ),
    },
    "./model/Qwen3-8B": {
        "name": "Qwen3-8B",
        "thinking": SamplingConfig(
            temperature=0.6, top_p=0.95, top_k=20,
            presence_penalty=1.5, repetition_penalty=1.0,
            max_tokens=32768, enable_thinking=True,
        ),
        "non_thinking": SamplingConfig(
            # Official Qwen3 non-thinking defaults (technical report + HF model card)
            temperature=0.7, top_p=0.8, top_k=20,
            presence_penalty=0.0, repetition_penalty=1.0,
            max_tokens=8192, enable_thinking=False,
        ),
    },
    # --- Gemma-3 models (non_thinking only, no thinking mode) ---
    "./model/gemma-3-1b-it": {
        "name": "gemma-3-1b-it",
        "non_thinking": SamplingConfig(
            temperature=1.0, top_p=0.95, top_k=64,
            presence_penalty=0.0, repetition_penalty=1.0,
            max_tokens=8192, enable_thinking=False,
        ),
    },
    "./model/gemma-3-4b-it": {
        "name": "gemma-3-4b-it",
        "non_thinking": SamplingConfig(
            # Official Gemma-3 defaults (HF model card: T=1.0, top_p=0.95, top_k=64)
            temperature=1.0, top_p=0.95, top_k=64,
            presence_penalty=0.0, repetition_penalty=1.0,
            max_tokens=8192, enable_thinking=False,
        ),
    },
    "./model/gemma-3-12b-it": {
        "name": "gemma-3-12b-it",
        "non_thinking": SamplingConfig(
            # Official Gemma-3 defaults (HF model card: T=1.0, top_p=0.95, top_k=64)
            temperature=1.0, top_p=0.95, top_k=64,
            presence_penalty=0.0, repetition_penalty=1.0,
            max_tokens=8192, enable_thinking=False,
        ),
    },
    # --- Llama models (non_thinking only) ---
    "./model/Llama-3.2-1B-Instruct": {
        "name": "Llama-3.2-1B-Instruct",
        "non_thinking": SamplingConfig(
            # Meta official generation defaults: T=0.6, top_p=0.9, no top_k
            temperature=0.6, top_p=0.9, top_k=-1,
            presence_penalty=0.0, repetition_penalty=1.0,
            max_tokens=8192, enable_thinking=False,
        ),
    },
    "./model/Llama-3.2-3B-Instruct": {
        "name": "Llama-3.2-3B-Instruct",
        "non_thinking": SamplingConfig(
            # Meta official generation defaults: T=0.6, top_p=0.9, no top_k
            temperature=0.6, top_p=0.9, top_k=-1,
            presence_penalty=0.0, repetition_penalty=1.0,
            max_tokens=8192, enable_thinking=False,
        ),
    },
    "./model/Llama-3.1-8B-Instruct": {
        "name": "Llama-3.1-8B-Instruct",
        "non_thinking": SamplingConfig(
            # Meta official (model card): T=0.6, top_p=0.9, no top_k
            temperature=0.6, top_p=0.9, top_k=-1,
            presence_penalty=0.0, repetition_penalty=1.0,
            max_tokens=8192, enable_thinking=False,
        ),
    },
    # --- Legacy models (kept for reference / comparison) ---
    "Qwen/Qwen2.5-7B-Instruct": {
        "name": "Qwen2.5-7B-Instruct",
        "non_thinking": SamplingConfig(
            temperature=0.7, top_p=0.8, top_k=20,
            presence_penalty=0.0, repetition_penalty=1.05,
            max_tokens=1024, enable_thinking=False,
        ),
    },
    "Qwen/Qwen2.5-Math-7B-Instruct": {
        "name": "Qwen2.5-Math-7B-Instruct",
        "non_thinking": SamplingConfig(
            temperature=0.7, top_p=0.8, top_k=20,
            presence_penalty=0.0, repetition_penalty=1.05,
            max_tokens=1024, enable_thinking=False,
            prompt_type="zeroshot",
        ),
    },
}

# Shorthand model aliases for CLI (--model argument)
# Maps alias → local path (downloaded via download_models.py → ./model/<name>/)
MODEL_ALIASES = {
    "qwen3-0.6b":      "./model/Qwen3-0.6B",
    "qwen3-4b":        "./model/Qwen3-4B",
    "qwen3-8b":        "./model/Qwen3-8B",
    "gemma-3-1b":      "./model/gemma-3-1b-it",
    "gemma-3-4b":      "./model/gemma-3-4b-it",
    "gemma-3-12b":     "./model/gemma-3-12b-it",
    "llama-3.2-1b":    "./model/Llama-3.2-1B-Instruct",
    "llama-3.2-3b":    "./model/Llama-3.2-3B-Instruct",
    "llama-3.1-8b":    "./model/Llama-3.1-8B-Instruct",
    "qwen2.5-7b":      "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-math-7b": "Qwen/Qwen2.5-Math-7B-Instruct",
}

PAPER_MODEL_ALIASES = [
    "qwen3-0.6b",
    "qwen3-4b",
    "qwen3-8b",
    "llama-3.2-1b",
    "llama-3.2-3b",
    "llama-3.1-8b",
    "gemma-3-4b",
]

DEFAULT_MODE = "non_thinking"


def _load_generation_config(model_path: str) -> dict:
    """Try to load generation_config.json from HuggingFace cache for a model."""
    try:
        from pathlib import Path
        import json
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(model_path, "generation_config.json")
        if cached and isinstance(cached, (str, Path)):
            with open(cached) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def resolve_model_path(model: str) -> str:
    """Resolve model alias/path to a canonical model path.

    Resolution order:
    1) Exact alias match in MODEL_ALIASES
    2) Case-insensitive alias match in MODEL_ALIASES
    3) Exact match in MODEL_CONFIGS keys
    4) Case-insensitive match in MODEL_CONFIGS keys
    5) Case-insensitive basename match against MODEL_CONFIGS keys
    6) Fallback to input as-is
    """
    if model is None:
        return model

    raw = str(model).strip()
    if not raw:
        return raw

    # Alias lookup (exact + case-insensitive)
    if raw in MODEL_ALIASES:
        return MODEL_ALIASES[raw]
    lowered = raw.lower()
    if lowered in MODEL_ALIASES:
        return MODEL_ALIASES[lowered]

    raw_norm = raw.rstrip("/")

    # Canonical model config path lookup
    if raw_norm in MODEL_CONFIGS:
        return raw_norm
    for cfg_key in MODEL_CONFIGS:
        if raw_norm.lower() == cfg_key.lower():
            return cfg_key

    # If user passed only model basename (or wrong-cased local path), map by basename.
    base = raw_norm.split("/")[-1]
    basename_matches = [
        cfg_key
        for cfg_key in MODEL_CONFIGS
        if cfg_key.rstrip("/").split("/")[-1].lower() == base.lower()
    ]
    if len(basename_matches) == 1:
        return basename_matches[0]

    return model


def get_default_mode(model_path: str = None) -> str:
    """Get the default mode for a model.

    Models with only one mode (e.g. Gemma3) return that mode.
    Models with both modes default to DEFAULT_MODE ('non_thinking').
    """
    resolved_model_path = resolve_model_path(model_path) if model_path else model_path
    if resolved_model_path and resolved_model_path in MODEL_CONFIGS:
        modes = [k for k in MODEL_CONFIGS[resolved_model_path] if k != "name"]
        if len(modes) == 1:
            return modes[0]
    return DEFAULT_MODE


def get_sampling_config(mode: str = DEFAULT_MODE, model_path: str = None) -> SamplingConfig:
    """Get sampling configuration for the specified mode and model.

    Priority:
    1. MODEL_CONFIGS entry for exact model_path match
    2. Auto-detect from model's generation_config.json (HF cache)
    3. Conservative generic defaults with warning
    """
    resolved_model_path = resolve_model_path(model_path) if model_path else model_path
    if resolved_model_path and resolved_model_path in MODEL_CONFIGS:
        model_cfg = MODEL_CONFIGS[resolved_model_path]
        if mode in model_cfg:
            return model_cfg[mode]
        available = [k for k in model_cfg if k != "name"]
        raise ValueError(
            f"Model {resolved_model_path} does not support mode '{mode}'. Available: {available}"
        )

    # Unknown model: try to auto-detect from generation_config.json
    if resolved_model_path:
        gen_cfg = _load_generation_config(resolved_model_path)
        if gen_cfg:
            print(f"[config] Auto-detected generation_config.json for {resolved_model_path}")
            return SamplingConfig(
                temperature=gen_cfg.get("temperature", 0.7),
                top_p=gen_cfg.get("top_p", 0.9),
                top_k=gen_cfg.get("top_k", -1),
                presence_penalty=0.0,
                repetition_penalty=gen_cfg.get("repetition_penalty", 1.0),
                max_tokens=8192,
                enable_thinking=False,
            )

    # Fallback: conservative generic defaults
    if resolved_model_path:
        print(f"[config] WARNING: Model '{resolved_model_path}' not in MODEL_CONFIGS. "
              f"Using conservative defaults (T=0.7). Add to MODEL_CONFIGS for optimal settings.")
    if mode == "thinking":
        return SamplingConfig(
            temperature=0.6, top_p=0.95, top_k=20,
            presence_penalty=1.5, repetition_penalty=1.0,
            max_tokens=32768, enable_thinking=True,
        )
    return SamplingConfig(
        temperature=0.7, top_p=0.9, top_k=-1,
        presence_penalty=0.0, repetition_penalty=1.0,
        max_tokens=8192, enable_thinking=False,
    )


def get_sampling_config_with_temperature(
    mode: str = DEFAULT_MODE,
    model_path: str = None,
    temperature: float = None,
) -> SamplingConfig:
    """Get sampling config, optionally overriding the temperature."""
    cfg = get_sampling_config(mode, model_path)
    if temperature is not None:
        from dataclasses import replace
        top_p = 1.0 if temperature == 0.0 else cfg.top_p
        cfg = replace(cfg, temperature=temperature, top_p=top_p)
    return cfg


# =============================================================================
# Model Info
# =============================================================================

def get_model_short_name(model_path: str) -> str:
    """Extract short model name from HuggingFace path."""
    resolved_model_path = resolve_model_path(model_path)
    if resolved_model_path in MODEL_CONFIGS:
        return MODEL_CONFIGS[resolved_model_path]["name"]
    return str(resolved_model_path).rstrip("/").split("/")[-1]


# =============================================================================
# Dataset Configuration
# =============================================================================

DATASET_NAME = "math500"
DEFAULT_SPLIT = "test"

# Dataset paths (relative to Cliff_Token_Analysis/ root)
# AIME paths are placeholders - fill in when data is available
DATASET_PATHS = {
    "gsm1k": {
        "test":  "./data/input/GSM1K_test.jsonl",
    },
    "gsm1k_100": {
        "test":  "./data/input/GSM1K_test_100.jsonl",
    },
    "math500": {
        "test":  "./data/input/MATH_test.jsonl",
    },
    "math500_100": {
        "test":  "./data/input/MATH_test_100.jsonl",
    },
    "aime24": {
        "test":  "./data/input/aime24.jsonl",
    },
    "aime25": {
        "test":  "./data/input/aime25.jsonl",
    },
    "gsm8k": {
        "test":  "./data/input/GSM8K_test.jsonl",
    },
    "gsm8k_1": {
        "test":  "./data/input/GSM8K_train_1.jsonl",
    },
    "gsm8k_2": {
        "test":  "./data/input/GSM8K_train_2.jsonl",
    },
    "gsm8k_3": {
        "test":  "./data/input/GSM8K_train_3.jsonl",
    },
}

# Maps dataset key to answer-extraction source name
DATASET_SOURCE_NAMES = {
    "gsm1k":     "GSM1K",
    "gsm1k_100": "GSM1K",
    "math500":   "MATH",
    "math500_100": "MATH",
    "aime24":  "AIME",
    "aime25":  "AIME",
    "gsm8k":       "GSM8K",
    "gsm8k_1": "GSM8K",
    "gsm8k_2": "GSM8K",
    "gsm8k_3": "GSM8K",
}


def get_dataset_path(dataset_name: str, split: str = DEFAULT_SPLIT) -> str:
    """Get dataset file path.

    Handles both canonical names (math500, gsm1k, aime24, aime25)
    and legacy shard names (e.g. GSM1K_shard0_...) for backward compatibility.
    """
    key = dataset_name.lower()
    if key in DATASET_PATHS:
        if split not in DATASET_PATHS[key]:
            raise ValueError(f"Unknown split '{split}' for dataset '{dataset_name}'")
        return DATASET_PATHS[key][split]
    # Fallback: ./data/input/{dataset_name}_{split}.jsonl
    return f"./data/input/{dataset_name}_{split}.jsonl"


def get_dataset_source_name(dataset_name: str) -> str:
    """Get answer-extraction source name for grading (e.g. 'MATH', 'GSM1K')."""
    key = dataset_name.lower()
    if key in DATASET_SOURCE_NAMES:
        return DATASET_SOURCE_NAMES[key]
    # Legacy: handle shard names
    if dataset_name.upper().startswith("MATH"):
        return "MATH"
    if dataset_name.upper().startswith("GSM1K"):
        return "GSM1K"
    if dataset_name.upper().startswith("AIME"):
        return "AIME"
    return dataset_name


# =============================================================================
# Token Limits (per dataset)
# =============================================================================

# Fast/default profile used in daily experiments.
_DATASET_MAX_TOKENS_DEFAULT = {
    "non_thinking": {
        "gsm1k":          1024,
        "math500":        2048,
        "aime24":         8192,
        "aime25":         8192,
    },
    "thinking": {
        "gsm1k":          16384,
        "math500":        16384,
        "aime24":         32768,
        "aime25":         32768,
    },
}

_DEFAULT_MAX_TOKENS_DEFAULT = {
    "non_thinking": 8192,
    "thinking": 32768,
}

# Paper profile for one-shot report runs.
# - General benchmarks: 32768
# - AIME'24/AIME'25:    38912
_DATASET_MAX_TOKENS_PAPER = {
    "non_thinking": {
        "gsm1k":   8192,
        "math500": 16384,
        "aime24":  32768,
        "aime25":  32768,
    },
    "thinking": {
        "gsm1k":   8192,
        "math500": 16384,
        "aime24":  32768,
        "aime25":  32768,
    },
}

_DEFAULT_MAX_TOKENS_PAPER = {
    "non_thinking": 32768,
    "thinking": 32768,
}

DATASET_MAX_TOKENS_PROFILES = {
    "default": _DATASET_MAX_TOKENS_DEFAULT,
    "paper": _DATASET_MAX_TOKENS_PAPER,
}

DEFAULT_MAX_TOKENS_PROFILES = {
    "default": _DEFAULT_MAX_TOKENS_DEFAULT,
    "paper": _DEFAULT_MAX_TOKENS_PAPER,
}

TOKEN_PROFILE_CHOICES = sorted(DATASET_MAX_TOKENS_PROFILES.keys())  # ["default", "paper"]

# Backward-compatible aliases used by existing call sites/constants references.
DATASET_MAX_TOKENS = DATASET_MAX_TOKENS_PROFILES["default"]
DEFAULT_MAX_TOKENS = DEFAULT_MAX_TOKENS_PROFILES["default"]


def get_max_tokens(
    dataset_name: str,
    mode: str = "non_thinking",
    token_profile: str = "default",
) -> int:
    """Get max_tokens for inference by dataset/mode/profile.

    token_profile:
      - default: current fast profile (backward compatible)
      - paper:   long-context profile for report-grade one-shot evaluation
    """
    if token_profile not in DATASET_MAX_TOKENS_PROFILES:
        allowed = sorted(DATASET_MAX_TOKENS_PROFILES.keys())
        raise ValueError(
            f"Unknown token_profile '{token_profile}'. "
            f"Allowed profiles: {allowed}"
        )

    tokens_map_all = DATASET_MAX_TOKENS_PROFILES[token_profile]
    default_map_all = DEFAULT_MAX_TOKENS_PROFILES[token_profile]
    key = dataset_name.lower()
    tokens_map = tokens_map_all.get(mode, tokens_map_all["non_thinking"])
    if key in tokens_map:
        return tokens_map[key]
    # Legacy shard names
    for k, v in tokens_map.items():
        if dataset_name.upper().startswith(k.upper()):
            return v
    return default_map_all.get(mode, default_map_all["non_thinking"])


# --- Rollout-specific token limits ---
# Rollout token limits are set short to match the actual token count of correct responses.
# Max for correct responses: AIME ~4200, MATH500 ~2048, GSM1K ~1024 tokens.
# No need to fill up to inference max_tokens, so a rollout-specific upper bound is used.
ROLLOUT_MAX_TOKENS = {
    "non_thinking": {
        "gsm1k":   1024,
        "math500": 2048,
        "aime24":  4096,
        "aime25":  4096,
    },
    "thinking": {
        "gsm1k":   16384,
        "math500": 16384,
        "aime24":  32768,
        "aime25":  32768,
    },
}

DEFAULT_ROLLOUT_MAX_TOKENS = {
    "non_thinking": 4096,
    "thinking": 32768,
}


def get_rollout_max_tokens(dataset_name: str, mode: str = "non_thinking") -> int:
    """Get max_tokens for rollout by dataset and mode."""
    key = dataset_name.lower()
    tokens_map = ROLLOUT_MAX_TOKENS.get(mode, ROLLOUT_MAX_TOKENS["non_thinking"])
    if key in tokens_map:
        return tokens_map[key]
    for k, v in tokens_map.items():
        if dataset_name.upper().startswith(k.upper()):
            return v
    return DEFAULT_ROLLOUT_MAX_TOKENS.get(mode, DEFAULT_ROLLOUT_MAX_TOKENS["non_thinking"])


# =============================================================================
# Rollout Configuration
# =============================================================================

ROLLOUT_SAMPLES = 64      # token-wise potential = 64-sample rollout
ROLLOUT_WINDOW_SIZE = 1   # token-wise (every single token)


# =============================================================================
# Cliff Token Thresholds
# =============================================================================

DEFAULT_CLIFF_THRESHOLD = 0.20   # Cliff token: potential drop >= 0.20
CRITICAL_TOKEN_THRESHOLD = 0.05  # Critical token: score=0 and all subsequent <= 0.05

# Alias for backward compatibility
DEFAULT_DROP_THRESHOLD = DEFAULT_CLIFF_THRESHOLD


# =============================================================================
# Batch Processing
# =============================================================================

BATCH_SIZE = 128
GPU_MEMORY_UTILIZATION = 0.95

MAX_NUM_SEQS = 384
MAX_NUM_BATCHED_TOKENS = 16384

GLOBAL_BATCH_SIZE = 64
MEMORY_LIMIT_REQUESTS = 50000
MAX_GRADING_WORKERS = 8
USE_OPTIMIZED_ROLLOUT = True
EARLY_TERMINATION_K = 20


# =============================================================================
# Output Directories
# =============================================================================

OUTPUT_DIR = "./output"


# =============================================================================
# Stop Tokens
# =============================================================================

STOP_TOKENS = ["\n\nQuestion:", ".\n\nQuestion:", "\n\nProblem:", ".\n\nProblem:"]


# =============================================================================
# Pass@K Evaluation
# =============================================================================

PASS_K_VALUES = [1, 2, 4, 8, 16, 32, 64]
