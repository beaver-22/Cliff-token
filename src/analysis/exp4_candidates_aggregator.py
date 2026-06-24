"""Exp2-2 analysis aggregator: Hazard profiling + Greedy recovery.

Inputs (must exist in run root):
  - cliff_instances.csv
  - candidate_results.csv

Outputs (under <run_dir>/<analysis_subdir>/):
  - tables/exp4_candidates_hazard_profile_macro.csv
  - figures/exp4_candidates_hazard_profile_macro.png
  - tables/exp4_candidates_representative_examples.csv
  - figures/type1_examples/*.png
  - figures/type2_examples/*.png
  - figures/type3_examples/*.png
  - tables/exp4_candidates_type3_greedy_recovery_points.csv
  - figures/exp4_candidates_type3_greedy_recovery_scatter.png
  - summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors


TYPE_ORDER = [
    ("type1", "deterministic failure (H low, greedy token)", "low-H + greedy"),
    ("type2", "ambiguous greedy (H high, greedy token)", "high-H + greedy"),
    ("type3", "sampling slip (H high, non-greedy token)", "high-H + non-greedy"),
]


def _parse_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}


def _to_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _to_int(v: str, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _sanitize_filename(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)[:180]


@dataclass
class CliffInfo:
    cliff_uid: str
    dataset: str
    path_id: str
    cliff_position: int
    cell_label: str
    entropy_at_t: float
    topk_prob_mass: float
    model: str = ""


@dataclass
class CliffMetric:
    cliff_uid: str
    sum_p_cliff: float
    sum_p_non_cliff: float
    n_cliff_tokens_top20: int
    topk_prob_mass_ref: float


def _load_and_validate(exp_dir: Path) -> Tuple[List[CliffInfo], Dict[str, List[Dict]], List[str]]:
    warnings: List[str] = []

    if not exp_dir.exists() or not exp_dir.is_dir():
        raise SystemExit(f"[exp4_candidates analysis] directory not found: {exp_dir}")

    cliffs_csv = exp_dir / "cliff_instances.csv"
    cands_csv = exp_dir / "candidate_results.csv"
    missing = [str(p.name) for p in (cliffs_csv, cands_csv) if not p.exists()]
    if missing:
        raise SystemExit(
            "[exp4_candidates analysis] merge seems incomplete. Missing required root files: "
            + ", ".join(missing)
            + f" (dir={exp_dir})"
        )

    cfg_path = exp_dir / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as exc:
            warnings.append(f"failed to parse config.json: {exc}")
            cfg = None
        if isinstance(cfg, dict):
            if cfg.get("merge_mode") is False and cfg.get("shard_index") is not None:
                raise SystemExit(
                    "[exp4_candidates analysis] detected shard output (merge_mode=false). "
                    "Please run analysis on merged exp4_candidates root output directory."
                )

    cliff_rows = _read_csv(cliffs_csv)
    if not cliff_rows:
        raise SystemExit("[exp4_candidates analysis] cliff_instances.csv is empty")
    cand_rows = _read_csv(cands_csv)
    if not cand_rows:
        raise SystemExit("[exp4_candidates analysis] candidate_results.csv is empty")

    cliffs: List[CliffInfo] = []
    for r in cliff_rows:
        cliffs.append(
            CliffInfo(
                cliff_uid=str(r["cliff_uid"]),
                dataset=str(r.get("dataset", "")),
                path_id=str(r.get("path_id", "")),
                cliff_position=_to_int(r.get("cliff_position", ""), 0),
                cell_label=str(r.get("cell_label", "")),
                entropy_at_t=_to_float(r.get("entropy_at_t", "0"), 0.0),
                topk_prob_mass=_to_float(r.get("topk_prob_mass", "0"), 0.0),
                model=str(r.get("model", "")),
            )
        )

    cands_by_uid: Dict[str, List[Dict]] = {}
    for r in cand_rows:
        uid = str(r.get("cliff_uid", ""))
        if not uid:
            continue
        row = {
            "cliff_uid": uid,
            "dataset": str(r.get("dataset", "")),
            "path_id": str(r.get("path_id", "")),
            "cliff_position": _to_int(r.get("cliff_position", ""), 0),
            "cell_label": str(r.get("cell_label", "")),
            "candidate_rank": _to_int(r.get("candidate_rank", ""), 10**9),
            "candidate_prob": _to_float(r.get("candidate_prob", "0"), 0.0),
            "candidate_logprob": _to_float(r.get("candidate_logprob", str(float("-inf"))), float("-inf")),
            "candidate_token_str": str(r.get("candidate_token_str", "")),
            "potential_t_candidate": _to_float(r.get("potential_t_candidate", "0"), 0.0),
            "is_candidate_cliff_stat": _parse_bool(r.get("is_candidate_cliff_stat", "")),
            "is_candidate_greedy": _parse_bool(r.get("is_candidate_greedy", "")),
            "is_candidate_selected_cliff": _parse_bool(r.get("is_candidate_selected_cliff", "")),
        }
        cands_by_uid.setdefault(uid, []).append(row)

    for uid in cands_by_uid:
        cands_by_uid[uid].sort(key=lambda x: (x["candidate_rank"], -x["candidate_prob"]))

    return cliffs, cands_by_uid, warnings


def _build_cliff_metrics(cliffs: List[CliffInfo], cands_by_uid: Dict[str, List[Dict]]) -> List[CliffMetric]:
    out: List[CliffMetric] = []
    for c in cliffs:
        rows = cands_by_uid.get(c.cliff_uid, [])
        # Cliff-mass definition for hazard profiling:
        # include statistically cliff-like alternatives AND the originally selected cliff token.
        sum_p_cliff = sum(
            r["candidate_prob"]
            for r in rows
            if (r["is_candidate_cliff_stat"] or r["is_candidate_selected_cliff"])
        )
        sum_p_non = sum(
            r["candidate_prob"]
            for r in rows
            if not (r["is_candidate_cliff_stat"] or r["is_candidate_selected_cliff"])
        )
        n_cliff = sum(
            1 for r in rows if (r["is_candidate_cliff_stat"] or r["is_candidate_selected_cliff"])
        )
        out.append(
            CliffMetric(
                cliff_uid=c.cliff_uid,
                sum_p_cliff=float(sum_p_cliff),
                sum_p_non_cliff=float(sum_p_non),
                n_cliff_tokens_top20=int(n_cliff),
                topk_prob_mass_ref=float(c.topk_prob_mass),
            )
        )
    return out


def _probabilities_at_temperature_top20_renorm(rows: List[Dict], temperature: float) -> List[float]:
    if not rows:
        return []
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")

    # T=0: deterministic argmax over the available top-20 candidates.
    if temperature == 0:
        best_idx = None
        best_lp = float("-inf")
        for idx, r in enumerate(rows):
            lp = float(r.get("candidate_logprob", float("-inf")))
            if math.isfinite(lp) and (best_idx is None or lp > best_lp):
                best_idx = idx
                best_lp = lp
        if best_idx is None:
            return [0.0 for _ in rows]
        out = [0.0 for _ in rows]
        out[best_idx] = 1.0
        return out

    scaled: List[float] = []
    for r in rows:
        lp = float(r.get("candidate_logprob", float("-inf")))
        if math.isfinite(lp):
            scaled.append(lp / temperature)
        else:
            scaled.append(float("-inf"))

    finite_vals = [v for v in scaled if math.isfinite(v)]
    if not finite_vals:
        return [0.0 for _ in rows]

    vmax = max(finite_vals)
    exps: List[float] = []
    for v in scaled:
        if math.isfinite(v):
            exps.append(math.exp(v - vmax))
        else:
            exps.append(0.0)
    denom = sum(exps)
    if denom <= 0:
        return [0.0 for _ in rows]
    return [x / denom for x in exps]


def _build_cliff_metrics_temp_top20_renorm(
    cliffs: List[CliffInfo],
    cands_by_uid: Dict[str, List[Dict]],
    temperature: float,
) -> List[CliffMetric]:
    out: List[CliffMetric] = []
    for c in cliffs:
        rows = cands_by_uid.get(c.cliff_uid, [])
        probs = _probabilities_at_temperature_top20_renorm(rows, temperature)
        sum_p_cliff = sum(
            p
            for p, r in zip(probs, rows)
            if (r["is_candidate_cliff_stat"] or r["is_candidate_selected_cliff"])
        )
        sum_p_non = sum(
            p
            for p, r in zip(probs, rows)
            if not (r["is_candidate_cliff_stat"] or r["is_candidate_selected_cliff"])
        )
        n_cliff = sum(
            1 for r in rows if (r["is_candidate_cliff_stat"] or r["is_candidate_selected_cliff"])
        )
        out.append(
            CliffMetric(
                cliff_uid=c.cliff_uid,
                sum_p_cliff=float(sum_p_cliff),
                sum_p_non_cliff=float(sum_p_non),
                n_cliff_tokens_top20=int(n_cliff),
                topk_prob_mass_ref=1.0,
            )
        )
    return out


def _temp_tag(temp: float) -> str:
    s = f"{temp:g}"
    return s.replace(".", "p")


def _temp_display_label(temp: float) -> str:
    return f"{temp:g}"


def _type_display_compact(type_key: str, type_name: str) -> str:
    compact = {
        "type1": "deterministic failure",
        "type2": "ambiguous greedy",
        "type3": "sampling slip",
    }
    return compact.get(type_key, type_name)


def _aggregate_macro_rows(
    cliffs: List[CliffInfo],
    cliff_metrics: List[CliffMetric],
) -> List[Dict]:
    metrics_by_uid = {m.cliff_uid: m for m in cliff_metrics}
    cliffs_by_type: Dict[str, List[CliffInfo]] = {key: [] for key, _, _ in TYPE_ORDER}
    total_cliffs = len(cliffs)

    cell_to_key = {cell: key for key, _, cell in TYPE_ORDER}
    for c in cliffs:
        key = cell_to_key.get(c.cell_label)
        if key is not None:
            cliffs_by_type[key].append(c)

    rows: List[Dict] = []
    for key, type_name, cell_label in TYPE_ORDER:
        group = cliffs_by_type[key]
        n = len(group)
        sum_p_cliff_vals = [metrics_by_uid[g.cliff_uid].sum_p_cliff for g in group if g.cliff_uid in metrics_by_uid]
        sum_p_non_vals = [metrics_by_uid[g.cliff_uid].sum_p_non_cliff for g in group if g.cliff_uid in metrics_by_uid]
        n_cliff_token_vals = [
            metrics_by_uid[g.cliff_uid].n_cliff_tokens_top20 for g in group if g.cliff_uid in metrics_by_uid
        ]
        entropy_vals = [g.entropy_at_t for g in group]

        row = {
            "type_key": key,
            "type_name": type_name,
            "cell_label": cell_label,
            "n_cliffs": n,
            "total_cliffs_denominator": total_cliffs,
            "mean_sum_p_cliff": statistics.fmean(sum_p_cliff_vals) if sum_p_cliff_vals else 0.0,
            "mean_sum_p_non_cliff": statistics.fmean(sum_p_non_vals) if sum_p_non_vals else 0.0,
            "avg_num_cliffs_in_top20": statistics.fmean(n_cliff_token_vals) if n_cliff_token_vals else 0.0,
            "mean_entropy": statistics.fmean(entropy_vals) if entropy_vals else 0.0,
        }
        rows.append(row)
    return rows


def _make_macro_table(
    cliffs: List[CliffInfo],
    cliff_metrics: List[CliffMetric],
    out_csv: Path,
    out_png: Path,
) -> List[Dict]:
    rows = _aggregate_macro_rows(cliffs, cliff_metrics)
    if not rows:
        _write_csv(out_csv, [], fieldnames=[])
        return rows
    _write_csv(out_csv, rows, fieldnames=list(rows[0].keys()))

    fig, ax = plt.subplots(figsize=(14, 3.2))
    ax.axis("off")
    table_data = []
    for r in rows:
        table_data.append(
            [
                _type_display_compact(str(r["type_key"]), str(r["type_name"])),
                f"{r['mean_sum_p_cliff']:.6f}",
                f"{r['mean_sum_p_non_cliff']:.6f}",
                f"{r['avg_num_cliffs_in_top20']:.3f}",
                f"{r['mean_entropy']:.6f}",
            ]
        )
    col_labels = [
        "Type",
        "Mean ΣP(cliff)",
        "Mean ΣP(non_cliff)",
        "Average #cliffs in Top-20",
        "Mean Entropy",
    ]
    tab = ax.table(cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.5)
    tab.scale(1.0, 1.55)

    n_rows = len(table_data) + 1
    n_cols = len(col_labels)
    col_widths = [0.24, 0.19, 0.19, 0.20, 0.18]
    for rr in range(n_rows):
        for cc in range(n_cols):
            cell = tab[(rr, cc)]
            cell.get_text().set_wrap(True)
            cell.set_width(col_widths[cc])

    ax.set_title("Exp2-2 Hazard Profiling (Macro)", fontsize=12, pad=10)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return rows


def _make_macro_temp_compare_table(
    cliffs: List[CliffInfo],
    temp_metrics_map: Dict[float, List[CliffMetric]],
    out_csv: Path,
    out_png: Path,
    display_temps: List[float],
    recommended_temp: Optional[float] = None,
) -> List[Dict]:
    temp_agg_by_temp: Dict[float, Dict[str, Dict]] = {}
    for temp in display_temps:
        temp_rows = _aggregate_macro_rows(cliffs, temp_metrics_map[temp])
        temp_agg_by_temp[temp] = {str(r["type_key"]): r for r in temp_rows}

    rows: List[Dict] = []
    for key, type_name, _cell_label in TYPE_ORDER:
        base_n = int(temp_agg_by_temp[display_temps[0]].get(key, {}).get("n_cliffs", 0)) if display_temps else 0
        row = {
            "type_name": _type_display_compact(key, type_name),
            "n_cliffs": base_n,
        }
        for temp in display_temps:
            tag = _temp_tag(temp)
            row[f"mean_sum_p_cliff_T{tag}_top20_renorm"] = float(
                temp_agg_by_temp[temp].get(key, {}).get("mean_sum_p_cliff", 0.0)
            )
        rows.append(row)

    if rows:
        _write_csv(out_csv, rows, fieldnames=list(rows[0].keys()))

    fig_w = max(8.0, 3.0 + 1.8 * len(display_temps))
    fig, ax = plt.subplots(figsize=(fig_w, 3.0))
    ax.axis("off")

    def _label_for_temp(t: float) -> str:
        label = f"T={_temp_display_label(t)} ΣP(cliff)"
        if recommended_temp is not None and math.isclose(t, recommended_temp, rel_tol=1e-6, abs_tol=1e-9):
            label += "\n(recommended)"
        return label

    col_labels = ["Type"] + [_label_for_temp(t) for t in display_temps]

    table_data = []
    for r in rows:
        vals = [str(r["type_name"])]
        for t in display_temps:
            vals.append(f"{r[f'mean_sum_p_cliff_T{_temp_tag(t)}_top20_renorm']:.6f}")
        table_data.append(vals)

    tab = ax.table(cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.4)
    tab.scale(1.0, 1.55)

    n_rows = len(table_data) + 1
    n_cols = len(col_labels)
    type_w = 0.20
    other_w = (1.0 - type_w) / max(1, (n_cols - 1))
    for rr in range(n_rows):
        for cc in range(n_cols):
            cell = tab[(rr, cc)]
            cell.get_text().set_wrap(True)
            cell.set_width(type_w if cc == 0 else other_w)

    ax.set_title("Exp2-2 ΣP(cliff) by Temperature", fontsize=12, pad=10)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)

    return rows


def _make_cliff_mass_temp_sweep(
    cliffs: List[CliffInfo],
    temp_metrics_map: Dict[float, List[CliffMetric]],
    out_csv: Path,
    out_png: Path,
) -> List[Dict]:
    temps_sorted = sorted(temp_metrics_map.keys())
    temp_agg_by_temp: Dict[float, Dict[str, Dict]] = {}
    for temp in temps_sorted:
        temp_rows = _aggregate_macro_rows(cliffs, temp_metrics_map[temp])
        temp_agg_by_temp[temp] = {str(r["type_key"]): r for r in temp_rows}

    rows: List[Dict] = []
    for key, type_name, _cell_label in TYPE_ORDER:
        row = {
            "type_name": _type_display_compact(key, type_name),
            "n_cliffs": int(temp_agg_by_temp[temps_sorted[0]].get(key, {}).get("n_cliffs", 0)) if temps_sorted else 0,
        }
        for temp in temps_sorted:
            tag = _temp_tag(temp)
            row[f"mean_sum_p_cliff_T{tag}_top20_renorm"] = float(
                temp_agg_by_temp[temp].get(key, {}).get("mean_sum_p_cliff", 0.0)
            )
        rows.append(row)

    if rows:
        _write_csv(out_csv, rows, fieldnames=list(rows[0].keys()))

    fig_w = max(12.0, 3.2 + 1.5 * (2 + len(temps_sorted)))
    fig, ax = plt.subplots(figsize=(fig_w, 3.1))
    ax.axis("off")

    col_labels = ["Type", "n_cliffs"] + [f"T={_temp_display_label(t)} ΣP(cliff)" for t in temps_sorted]
    table_data = []
    for r in rows:
        vals = [
            str(r["type_name"]),
            str(int(r["n_cliffs"])),
        ]
        for t in temps_sorted:
            vals.append(f"{r[f'mean_sum_p_cliff_T{_temp_tag(t)}_top20_renorm']:.6f}")
        table_data.append(vals)

    tab = ax.table(cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.6)
    tab.scale(1.0, 1.6)

    n_rows = len(table_data) + 1
    n_cols = len(col_labels)
    type_w = 0.20
    n_w = 0.08
    rest_w = (1.0 - type_w - n_w) / max(1, len(temps_sorted))
    for rr in range(n_rows):
        for cc in range(n_cols):
            cell = tab[(rr, cc)]
            cell.get_text().set_wrap(True)
            if cc == 0:
                cell.set_width(type_w)
            elif cc == 1:
                cell.set_width(n_w)
            else:
                cell.set_width(rest_w)

    ax.set_title("Exp2-2 Cliff Mass by Temperature (Top-20 Renorm)", fontsize=12, pad=10)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)

    return rows


def _pick_row(rows: List[Dict], *, flag: str) -> Optional[Dict]:
    picked = [r for r in rows if r.get(flag)]
    if not picked:
        return None
    picked.sort(key=lambda x: (x["candidate_rank"], -x["candidate_prob"]))
    return picked[0]


def _pick_rank(rows: List[Dict], rank: int) -> Optional[Dict]:
    picked = [r for r in rows if r["candidate_rank"] == rank]
    if not picked:
        return None
    picked.sort(key=lambda x: -x["candidate_prob"])
    return picked[0]


def _compute_type_score(type_key: str, rows: List[Dict]) -> Dict:
    selected = _pick_row(rows, flag="is_candidate_selected_cliff")
    greedy = _pick_row(rows, flag="is_candidate_greedy")
    rank1 = _pick_rank(rows, 1)
    rank2 = _pick_rank(rows, 2)
    rank3 = _pick_rank(rows, 3)

    selected_pot = selected["potential_t_candidate"] if selected else 0.0
    selected_rank = selected["candidate_rank"] if selected else None
    selected_prob = selected["candidate_prob"] if selected else None
    greedy_pot = greedy["potential_t_candidate"] if greedy else None
    greedy_rank = greedy["candidate_rank"] if greedy else None

    score = -10**9
    best_alt_rank = None
    best_alt_pot = None
    score_detail = ""

    if type_key == "type1":
        # Top-tier vs next-tier prob gap. Under competition tie-aware rank,
        # rank==1 captures tied-top tokens and rank>1 captures strictly
        # lower-prob tokens. Group-split by rank for stability (integer
        # compare, no float EPS threshold).
        top_tier = [r for r in rows if r["candidate_rank"] == 1]
        non_top = [r for r in rows if r["candidate_rank"] > 1]
        p1 = max((r["candidate_prob"] for r in top_tier), default=0.0)
        p2 = max((r["candidate_prob"] for r in non_top), default=p1)
        score = (p1 - p2) + (1.0 - selected_pot)
        score_detail = f"(top_tier_prob-next_tier_prob)+(1-selected_potential)=({p1:.6f}-{p2:.6f})+(1-{selected_pot:.6f})"
    elif type_key == "type2":
        # Near-top non-selected alternatives. Competition rank's tied-top
        # tokens share rank=1, so the old `rank in (2, 3)` filter would
        # drop them. Sort by (rank asc, prob desc) and take top-3 to
        # capture intent: "top few by rank tier, ties broken by prob".
        non_selected = [r for r in rows if not r["is_candidate_selected_cliff"]]
        non_selected.sort(key=lambda r: (r["candidate_rank"], -r["candidate_prob"]))
        preferred = non_selected[:3]
        if preferred:
            best_alt = max(preferred, key=lambda x: x["potential_t_candidate"])
        else:
            best_alt = None
        if best_alt is not None:
            best_alt_rank = best_alt["candidate_rank"]
            best_alt_pot = best_alt["potential_t_candidate"]
            score = best_alt_pot - selected_pot
            score_detail = f"best_alt_potential-selected_potential={best_alt_pot:.6f}-{selected_pot:.6f}"
        else:
            score = -10**9
            score_detail = "no non-selected alternatives found"
    elif type_key == "type3":
        if greedy is not None:
            score = greedy_pot - selected_pot
            score_detail = f"greedy_potential-selected_potential={greedy_pot:.6f}-{selected_pot:.6f}"
        else:
            score = -10**9
            score_detail = "missing greedy candidate row"

    return {
        "selection_score": float(score),
        "selected_rank": selected_rank,
        "selected_prob": selected_prob,
        "selected_potential": selected_pot,
        "greedy_rank": greedy_rank,
        "greedy_potential": greedy_pot,
        "best_alt_rank": best_alt_rank,
        "best_alt_potential": best_alt_pot,
        "score_detail": score_detail,
    }


def _draw_top20_chart(
    type_name: str,
    cliff_uid: str,
    cliff: CliffInfo,
    rows: List[Dict],
    out_png: Path,
) -> None:
    rows_sorted = sorted(rows, key=lambda x: x["candidate_rank"])
    ranks = [r["candidate_rank"] for r in rows_sorted]
    probs = [r["candidate_prob"] for r in rows_sorted]

    sel_row = _pick_row(rows_sorted, flag="is_candidate_selected_cliff")
    sel_rank = sel_row["candidate_rank"] if sel_row is not None else None

    # Any candidate that is itself classified as a cliff token gets the
    # red fill — not just the one that was actually sampled.
    def _is_cliff(row) -> bool:
        v = row.get("is_candidate_cliff_stat")
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes")
        return bool(v)

    cliff_ranks = {r["candidate_rank"] for r in rows_sorted if _is_cliff(r)}
    if sel_rank is not None:
        cliff_ranks.add(sel_rank)

    BAR_COLOR = "#B0BEC5"   # neutral gray for non-cliff bars
    CLIFF_RED = "#E53935"   # cliff token fill
    bar_colors = [CLIFF_RED if r in cliff_ranks else BAR_COLOR for r in ranks]
    # The selected (actually sampled) cliff token keeps the thick outline
    # so it is still distinguishable when multiple cliff candidates exist.
    edge_colors = ["black" if r == sel_rank else "#455A64" for r in ranks]
    edge_widths = [2.0 if r == sel_rank else 0.35 for r in ranks]

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    bars = ax.bar(ranks, probs, color=bar_colors,
                  edgecolor=edge_colors, linewidth=edge_widths)
    ax.set_xlim(0.5, max(20, max(ranks) + 0.5))
    ax.set_ylim(0, 1.0)
    ax.set_xticks(list(range(1, 21)))
    ax.set_xlabel("Token Rank (Top-20)")
    ax.set_ylabel("Token Probability (p)")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)

    if cliff_ranks:
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=CLIFF_RED, edgecolor="#455A64",
                         linewidth=0.35, label="Cliff token")]
        if sel_rank is not None:
            handles.append(Patch(facecolor=CLIFF_RED, edgecolor="black",
                                 linewidth=2.0, label="Cliff token (sampled)"))
        ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=9)

    ax.set_title(
        f"{type_name}\n{cliff.dataset} | pos={cliff.cliff_position} | uid={cliff_uid}",
        fontsize=10,
    )

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _build_taxonomy_boxplot(
    cliffs: List[CliffInfo],
    metrics_by_uid: Dict[str, "CliffMetric"],
    out_png: Path,
    value_fn,
    y_label: str,
    title: str,
    y_limits: Optional[Tuple[float, float]] = None,
) -> Dict:
    """Box plot of a per-cliff-instance metric, grouped by (taxonomy, model).

    One figure, x = taxonomy (deterministic / uncertain / sampled_off), hue = model, y = value_fn(metric).
    Returns a small summary dict with per-(model, type) counts and medians.
    """
    cell_to_key = {cell: (key, name) for key, name, cell in TYPE_ORDER}
    records: List[Tuple[str, str, str, float]] = []
    for c in cliffs:
        m = metrics_by_uid.get(c.cliff_uid)
        if m is None:
            continue
        mapped = cell_to_key.get(c.cell_label)
        if mapped is None:
            continue  # skip low-H + non-greedy (not in 3-class taxonomy)
        type_key, type_name = mapped
        model_name = c.model or "(unknown)"
        records.append((model_name, type_key, type_name, float(value_fn(m))))

    out_png.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.text(0.5, 0.5, "No data for taxonomy boxplot",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(out_png, dpi=170, bbox_inches="tight")
        plt.close(fig)
        return {"n_total": 0}

    # Ordered categories and models (stable)
    type_keys = [k for k, _, _ in TYPE_ORDER]
    type_name_by_key = {k: n for k, n, _ in TYPE_ORDER}
    type_abbr_by_key = {
        "type1": "Deterministic",
        "type2": "Uncertain",
        "type3": "Sampled-off",
    }
    models_seen: List[str] = []
    for mn, *_ in records:
        if mn not in models_seen:
            models_seen.append(mn)

    # Bucket: data[type_key][model] = list of values
    data: Dict[str, Dict[str, List[float]]] = {
        k: {m: [] for m in models_seen} for k in type_keys
    }
    for m, tk, _, v in records:
        data[tk][m].append(v)

    # Layout: for each category, draw one box per model; space categories apart.
    n_models = len(models_seen)
    box_width = 0.72 / max(n_models, 1)
    group_gap = 1.0
    model_cmap = plt.cm.tab10(np.linspace(0, 1, max(n_models, 2))) if n_models > 1 \
        else [(0.90, 0.30, 0.24, 1.0)]

    fig, ax = plt.subplots(figsize=(max(7.0, 2.6 * len(type_keys) + 1.4 * n_models), 5.2))
    xtick_positions: List[float] = []
    xtick_labels: List[str] = []
    legend_patches = []
    from matplotlib.patches import Patch

    for ti, tk in enumerate(type_keys):
        group_center = ti * group_gap * (1 + n_models * box_width * 0.5)
        # Simpler: fixed centre per category, boxes offset around it
        group_center = ti * 1.6
        xtick_positions.append(group_center)
        xtick_labels.append(type_abbr_by_key.get(tk, type_name_by_key[tk]))
        for mi, m in enumerate(models_seen):
            vals = data[tk][m]
            if not vals:
                continue
            offset = (mi - (n_models - 1) / 2) * box_width
            pos = group_center + offset
            bp = ax.boxplot(
                [vals], positions=[pos], widths=box_width * 0.9,
                patch_artist=True, showmeans=True,
                meanprops={"marker": "D", "markerfacecolor": "white",
                           "markeredgecolor": "black", "markersize": 5},
                medianprops={"color": "black", "linewidth": 1.2},
                boxprops={"facecolor": model_cmap[mi], "alpha": 0.75,
                          "edgecolor": "black", "linewidth": 0.9},
                whiskerprops={"color": "black", "linewidth": 0.8},
                capprops={"color": "black", "linewidth": 0.8},
                flierprops={"marker": "o", "markerfacecolor": model_cmap[mi],
                            "markeredgecolor": "black", "markersize": 3,
                            "alpha": 0.6},
            )

    for mi, m in enumerate(models_seen):
        legend_patches.append(Patch(facecolor=model_cmap[mi], alpha=0.75,
                                    edgecolor="black", label=m))

    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, fontsize=10)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    else:
        ax.set_ylim(bottom=0)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.6)
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9, frameon=True)

    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for tk in type_keys:
        summary[tk] = {}
        for m in models_seen:
            vals = data[tk][m]
            if not vals:
                continue
            summary[tk][m] = {
                "n": float(len(vals)),
                "mean": float(statistics.fmean(vals)),
                "median": float(statistics.median(vals)),
            }
    return summary


def _build_representatives(
    cliffs: List[CliffInfo],
    cands_by_uid: Dict[str, List[Dict]],
    figures_dir: Path,
    table_path: Path,
) -> Tuple[List[Dict], List[str]]:
    warnings: List[str] = []
    cliff_map = {c.cliff_uid: c for c in cliffs}
    rows_out: List[Dict] = []

    for type_key, type_name, cell_label in TYPE_ORDER:
        group_cliffs = [c for c in cliffs if c.cell_label == cell_label]
        candidates: List[Dict] = []
        for c in group_cliffs:
            crow = cands_by_uid.get(c.cliff_uid, [])
            if not crow:
                continue
            score_info = _compute_type_score(type_key, crow)
            candidates.append(
                {
                    "cliff_uid": c.cliff_uid,
                    "dataset": c.dataset,
                    "path_id": c.path_id,
                    "cliff_position": c.cliff_position,
                    "cell_label": c.cell_label,
                    "type_key": type_key,
                    "type_name": type_name,
                    **score_info,
                }
            )

        candidates.sort(key=lambda x: (-x["selection_score"], x["cliff_uid"]))
        selected = candidates[:3]
        if len(selected) < 3:
            warnings.append(
                f"{type_key}: only {len(selected)} representative samples available (requested 3)"
            )

        subdir = figures_dir / f"{type_key}_examples"
        for idx, rec in enumerate(selected, start=1):
            cliff_uid = rec["cliff_uid"]
            cinfo = cliff_map[cliff_uid]
            chart_rows = cands_by_uid[cliff_uid]
            fname = f"{type_key}_{idx:02d}_{_sanitize_filename(cliff_uid)}.png"
            out_png = subdir / fname
            _draw_top20_chart(type_name, cliff_uid, cinfo, chart_rows, out_png)

            rec = dict(rec)
            rec["figure_path"] = str(out_png)
            rows_out.append(rec)

    if rows_out:
        _write_csv(table_path, rows_out, fieldnames=list(rows_out[0].keys()))
    else:
        _write_csv(
            table_path,
            [],
            fieldnames=[
                "cliff_uid",
                "dataset",
                "path_id",
                "cliff_position",
                "cell_label",
                "type_key",
                "type_name",
                "selection_score",
                "selected_rank",
                "selected_prob",
                "selected_potential",
                "greedy_rank",
                "greedy_potential",
                "best_alt_rank",
                "best_alt_potential",
                "score_detail",
                "figure_path",
            ],
        )

    return rows_out, warnings


def _build_type3_recovery(
    cliffs: List[CliffInfo],
    cands_by_uid: Dict[str, List[Dict]],
    out_points_csv: Path,
    out_scatter_png: Path,
) -> Dict:
    cell_label = "high-H + non-greedy"
    type3_cliffs = [c for c in cliffs if c.cell_label == cell_label]

    points: List[Dict] = []
    for c in type3_cliffs:
        rows = cands_by_uid.get(c.cliff_uid, [])
        if not rows:
            continue
        selected = _pick_row(rows, flag="is_candidate_selected_cliff")
        greedy = _pick_row(rows, flag="is_candidate_greedy")
        if selected is None or greedy is None:
            continue

        x = selected["potential_t_candidate"]
        y = greedy["potential_t_candidate"]
        points.append(
            {
                "cliff_uid": c.cliff_uid,
                "dataset": c.dataset,
                "path_id": c.path_id,
                "cliff_position": c.cliff_position,
                "cell_label": c.cell_label,
                "selected_rank": selected["candidate_rank"],
                "selected_potential": x,
                "greedy_rank": greedy["candidate_rank"],
                "greedy_potential": y,
                "recovery_gap": y - x,
                "entropy_at_t": c.entropy_at_t,
            }
        )

    if points:
        _write_csv(out_points_csv, points, fieldnames=list(points[0].keys()))
    else:
        _write_csv(
            out_points_csv,
            [],
            fieldnames=[
                "cliff_uid",
                "dataset",
                "path_id",
                "cliff_position",
                "cell_label",
                "selected_rank",
                "selected_potential",
                "greedy_rank",
                "greedy_potential",
                "recovery_gap",
                "entropy_at_t",
            ],
        )

    fig, ax = plt.subplots(figsize=(7.8, 6.3))
    if points:
        xs = [p["selected_potential"] for p in points]
        ys = [p["greedy_potential"] for p in points]
        ents = [p["entropy_at_t"] for p in points]
        scatter = ax.scatter(xs, ys, c=ents, cmap="viridis", s=36, alpha=0.85, linewidths=0)
        cbar = fig.colorbar(scatter, ax=ax, pad=0.015)
        cbar.set_label("Entropy at cliff token")
    else:
        ax.text(0.5, 0.5, "No Type3 points available", ha="center", va="center", transform=ax.transAxes)

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="gray", label="y = x")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Selected non-greedy potential")
    ax.set_ylabel("Recovered greedy potential")
    ax.set_title("Exp2-2 Type3 Greedy Recovery")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    fig.tight_layout()
    out_scatter_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_scatter_png, dpi=170, bbox_inches="tight")
    plt.close(fig)

    gaps = [p["recovery_gap"] for p in points]
    selected_vals = [p["selected_potential"] for p in points]
    greedy_vals = [p["greedy_potential"] for p in points]
    summary = {
        "n_points": len(points),
        "mean_recovery_gap": statistics.fmean(gaps) if gaps else 0.0,
        "median_recovery_gap": statistics.median(gaps) if gaps else 0.0,
        "pct_greedy_better": (100.0 * sum(1 for g in gaps if g > 0) / len(gaps)) if gaps else 0.0,
        "mean_selected_potential": statistics.fmean(selected_vals) if selected_vals else 0.0,
        "mean_greedy_potential": statistics.fmean(greedy_vals) if greedy_vals else 0.0,
    }
    return summary


def _mass_consistency(cliff_metrics: List[CliffMetric]) -> Dict:
    errors = []
    for m in cliff_metrics:
        approx = m.sum_p_cliff + m.sum_p_non_cliff
        errors.append(abs(approx - m.topk_prob_mass_ref))
    if not errors:
        return {
            "n_cliffs_checked": 0,
            "top20_mass_mean_abs_error": 0.0,
            "top20_mass_max_abs_error": 0.0,
            "top20_mass_rms_error": 0.0,
        }
    rms = math.sqrt(sum(e * e for e in errors) / len(errors))
    return {
        "n_cliffs_checked": len(errors),
        "top20_mass_mean_abs_error": float(statistics.fmean(errors)),
        "top20_mass_max_abs_error": float(max(errors)),
        "top20_mass_rms_error": float(rms),
    }


def run(
    exp4_output_dir: Path,
    analysis_subdir: str = "analysis",
    target_temps: Optional[List[float]] = None,
) -> Dict:
    cliffs, cands_by_uid, warnings = _load_and_validate(exp4_output_dir)
    cliff_metrics = _build_cliff_metrics(cliffs, cands_by_uid)

    temps_raw = target_temps if target_temps is not None else [0.0, 0.7, 1.0, 2.0, 5.0]
    temps_clean: List[float] = []
    for t in temps_raw:
        try:
            ft = float(t)
        except Exception:
            warnings.append(f"invalid target temperature ignored: {t}")
            continue
        if ft < 0:
            warnings.append(f"negative target temperature ignored: {t}")
            continue
        temps_clean.append(ft)
    if 0.7 not in temps_clean:
        temps_clean.append(0.7)
    temps_clean = sorted(set(temps_clean))

    analysis_root = exp4_output_dir / analysis_subdir
    tables_dir = analysis_root / "tables"
    figures_dir = analysis_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    macro_rows = _make_macro_table(
        cliffs=cliffs,
        cliff_metrics=cliff_metrics,
        out_csv=tables_dir / "exp4_candidates_hazard_profile_macro.csv",
        out_png=figures_dir / "exp4_candidates_hazard_profile_macro.png",
    )

    temp_metrics_map: Dict[float, List[CliffMetric]] = {}
    for temp in temps_clean:
        temp_metrics_map[temp] = _build_cliff_metrics_temp_top20_renorm(
            cliffs=cliffs,
            cands_by_uid=cands_by_uid,
            temperature=temp,
        )

    # Resolve model's recommended temperature from config.json + src.config
    recommended_temp: float = 0.7
    cfg_path = exp4_output_dir / "config.json"
    if cfg_path.exists():
        try:
            cfg_dict = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as exc:
            cfg_dict = {}
            warnings.append(f"failed to parse config.json for recommended temp: {exc}")
        model_alias = str(cfg_dict.get("model", "") or "")
        cfg_temp = cfg_dict.get("temperature", None)
        mode = str(cfg_dict.get("mode", "non_thinking") or "non_thinking")
        try:
            if cfg_temp is not None:
                recommended_temp = float(cfg_temp)
            elif model_alias:
                from src import config as _project_config
                resolved = _project_config.resolve_model_path(model_alias)
                scfg = _project_config.get_sampling_config(mode=mode, model_path=resolved)
                recommended_temp = float(scfg.temperature)
        except Exception as exc:
            warnings.append(f"failed to resolve recommended temperature: {exc}")

    # Macro temp-compare uses exactly 3 columns: [recommended, 0.1, 2.0]
    compare_temps_raw = [recommended_temp, 0.1, 2.0]
    compare_temps: List[float] = []
    _seen: set = set()
    for t in compare_temps_raw:
        key = round(float(t), 6)
        if key not in _seen:
            _seen.add(key)
            compare_temps.append(float(t))
    for t in compare_temps:
        if t not in temp_metrics_map:
            temp_metrics_map[t] = _build_cliff_metrics_temp_top20_renorm(
                cliffs=cliffs,
                cands_by_uid=cands_by_uid,
                temperature=t,
            )

    macro_temp_compare_rows: List[Dict] = []
    cliff_mass_temp_rows: List[Dict] = []
    if temp_metrics_map:
        macro_temp_compare_rows = _make_macro_temp_compare_table(
            cliffs=cliffs,
            temp_metrics_map=temp_metrics_map,
            out_csv=tables_dir / "exp4_candidates_hazard_profile_macro_temp_compare.csv",
            out_png=figures_dir / "exp4_candidates_hazard_profile_macro_temp_compare.png",
            display_temps=compare_temps,
            recommended_temp=recommended_temp,
        )
        cliff_mass_temp_rows = _make_cliff_mass_temp_sweep(
            cliffs=cliffs,
            temp_metrics_map=temp_metrics_map,
            out_csv=tables_dir / "exp4_candidates_cliff_mass_temp_sweep.csv",
            out_png=figures_dir / "exp4_candidates_cliff_mass_temp_sweep.png",
        )

    reps_rows, rep_warnings = _build_representatives(
        cliffs=cliffs,
        cands_by_uid=cands_by_uid,
        figures_dir=figures_dir,
        table_path=tables_dir / "exp4_candidates_representative_examples.csv",
    )
    warnings.extend(rep_warnings)

    type3_summary = _build_type3_recovery(
        cliffs=cliffs,
        cands_by_uid=cands_by_uid,
        out_points_csv=tables_dir / "exp4_candidates_type3_greedy_recovery_points.csv",
        out_scatter_png=figures_dir / "exp4_candidates_type3_greedy_recovery_scatter.png",
    )

    metrics_by_uid = {m.cliff_uid: m for m in cliff_metrics}
    sum_p_cliff_boxplot_summary = _build_taxonomy_boxplot(
        cliffs=cliffs,
        metrics_by_uid=metrics_by_uid,
        out_png=figures_dir / "exp4_candidates_sum_p_cliff_boxplot.png",
        value_fn=lambda m: m.sum_p_cliff,
        y_label="sum(P(cliff))  (top-20 cliff-token probability mass)",
        title="Distribution of sum(P(cliff)) per cliff instance — by taxonomy",
        y_limits=(0.0, 1.0),
    )
    n_cliff_tokens_boxplot_summary = _build_taxonomy_boxplot(
        cliffs=cliffs,
        metrics_by_uid=metrics_by_uid,
        out_png=figures_dir / "exp4_candidates_n_cliff_tokens_boxplot.png",
        value_fn=lambda m: m.n_cliff_tokens_top20,
        y_label="n(cliff tokens)  (top-20 candidates flagged as cliff)",
        title="Distribution of n(cliff tokens) per cliff instance — by taxonomy",
    )

    cell_counts: Dict[str, int] = {}
    for c in cliffs:
        cell_counts[c.cell_label] = cell_counts.get(c.cell_label, 0) + 1

    candidate_group_counts: Dict[str, int] = {}
    for uid, rows in cands_by_uid.items():
        if not rows:
            continue
        label = rows[0].get("cell_label", "")
        candidate_group_counts[label] = candidate_group_counts.get(label, 0) + 1

    for label, n_cliffs in cell_counts.items():
        n_groups = candidate_group_counts.get(label, 0)
        if n_cliffs != n_groups:
            warnings.append(
                f"cell mismatch for '{label}': cliffs={n_cliffs}, candidate_groups={n_groups}"
            )

    rep_counts: Dict[str, int] = {}
    for r in reps_rows:
        rep_counts[r["type_key"]] = rep_counts.get(r["type_key"], 0) + 1

    artifacts = {
        "macro_csv": str(tables_dir / "exp4_candidates_hazard_profile_macro.csv"),
        "macro_png": str(figures_dir / "exp4_candidates_hazard_profile_macro.png"),
        "representatives_csv": str(tables_dir / "exp4_candidates_representative_examples.csv"),
        "recovery_points_csv": str(tables_dir / "exp4_candidates_type3_greedy_recovery_points.csv"),
        "recovery_scatter_png": str(figures_dir / "exp4_candidates_type3_greedy_recovery_scatter.png"),
        "sum_p_cliff_boxplot_png": str(figures_dir / "exp4_candidates_sum_p_cliff_boxplot.png"),
        "n_cliff_tokens_boxplot_png": str(figures_dir / "exp4_candidates_n_cliff_tokens_boxplot.png"),
    }
    if temp_metrics_map:
        artifacts["macro_temp_compare_csv"] = str(tables_dir / "exp4_candidates_hazard_profile_macro_temp_compare.csv")
        artifacts["macro_temp_compare_png"] = str(figures_dir / "exp4_candidates_hazard_profile_macro_temp_compare.png")
        artifacts["cliff_mass_temp_sweep_csv"] = str(tables_dir / "exp4_candidates_cliff_mass_temp_sweep.csv")
        artifacts["cliff_mass_temp_sweep_png"] = str(figures_dir / "exp4_candidates_cliff_mass_temp_sweep.png")

    summary = {
        "input_dir": str(exp4_output_dir),
        "analysis_dir": str(analysis_root),
        "n_total_cliffs": len(cliffs),
        "n_total_candidate_groups": len(cands_by_uid),
        "cell_counts": cell_counts,
        "candidate_group_counts": candidate_group_counts,
        "macro_rows": macro_rows,
        "temperature_compare": {
            "target_temps": temps_clean,
            "probability_definition": "Top-20 renormalized ΣP(cliff) from candidate_logprob/temperature",
            "recommended_temp": recommended_temp,
            "macro_compare_display_temps": compare_temps,
            "rows": macro_temp_compare_rows,
            "cliff_mass_only_rows": cliff_mass_temp_rows,
        },
        "representative_counts": rep_counts,
        "type3_recovery": type3_summary,
        "consistency_checks": _mass_consistency(cliff_metrics),
        "warnings": warnings,
        "artifacts": artifacts,
    }
    (analysis_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze exp4_candidates merged outputs")
    parser.add_argument("exp4_output_dir", help="exp4 output directory")
    parser.add_argument("--analysis_subdir", default="analysis", help="Subdirectory name for analysis outputs")
    parser.add_argument(
        "--target_temps",
        default="0,0.7,1,2,5",
        help="Comma-separated what-if temperatures (Top-20 renormalized from candidate_logprob). Supports T=0 (argmax over top-20). Example: 0,0.7,1,2,5",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    temps = []
    for x in str(args.target_temps).split(","):
        x = x.strip()
        if not x:
            continue
        try:
            temps.append(float(x))
        except Exception:
            temps.append(x)

    summary = run(
        Path(args.exp4_output_dir),
        analysis_subdir=args.analysis_subdir,
        target_temps=temps,
    )
    print("=" * 60)
    print("Exp2-2 Analysis Complete")
    print("=" * 60)
    print(f"Input dir:      {summary['input_dir']}")
    print(f"Analysis dir:   {summary['analysis_dir']}")
    print(f"Total cliffs:   {summary['n_total_cliffs']}")
    print(f"Type3 points:   {summary['type3_recovery']['n_points']}")
    tcmp = summary.get("temperature_compare", {})
    print(f"Temp compare:   {tcmp.get('target_temps', [])}")
    print(f"Warnings:       {len(summary['warnings'])}")
    for w in summary["warnings"]:
        print(f"  - {w}")
    print(f"Summary JSON:   {summary['analysis_dir']}/summary.json")


if __name__ == "__main__":
    main()
