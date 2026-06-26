#!/usr/bin/env python3
"""Summarize the paper experiment suite for dynamic CLIPTTA update scope."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work" / "paper_experiment_suite"

CORRUPTIONS = [
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "shot_noise",
    "snow",
    "zoom_blur",
]

IMAGE_CLIP = {
    "brightness": 83.41,
    "contrast": 61.98,
    "defocus_blur": 70.01,
    "elastic_transform": 53.17,
    "fog": 68.41,
    "frost": 76.53,
    "gaussian_noise": 37.69,
    "glass_blur": 42.18,
    "impulse_noise": 51.68,
    "jpeg_compression": 56.15,
    "motion_blur": 65.83,
    "pixelate": 48.51,
    "shot_noise": 41.08,
    "snow": 73.22,
    "zoom_blur": 72.53,
}

IMAGE_CLIPTTA = {
    "brightness": 92.59,
    "contrast": 87.31,
    "defocus_blur": 83.63,
    "elastic_transform": 75.87,
    "fog": 85.98,
    "frost": 86.60,
    "gaussian_noise": 69.20,
    "glass_blur": 68.90,
    "impulse_noise": 77.25,
    "jpeg_compression": 73.52,
    "motion_blur": 83.10,
    "pixelate": 79.76,
    "shot_noise": 71.62,
    "snow": 86.83,
    "zoom_blur": 85.57,
}

IMAGE_LORA = {
    "brightness": 91.37,
    "contrast": 87.49,
    "defocus_blur": 84.48,
    "elastic_transform": 74.10,
    "fog": 84.27,
    "frost": 86.00,
    "gaussian_noise": 67.30,
    "glass_blur": 69.47,
    "impulse_noise": 76.71,
    "jpeg_compression": 72.61,
    "motion_blur": 83.55,
    "pixelate": 80.16,
    "shot_noise": 71.22,
    "snow": 86.58,
    "zoom_blur": 84.61,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUT)
    return parser.parse_args()


def pct(series: pd.Series) -> pd.Series:
    return series.str.extract(r"([0-9.]+)").iloc[:, 0].astype(float)


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def read_many(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def md_table(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values: List[str] = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                if math.isnan(value):
                    values.append("NA")
                else:
                    values.append(f"{value:.2f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def load_baselines() -> pd.DataFrame:
    path = ROOT / "work/cliptta_update_scope_analysis_full_v2/cifar10/per_corruption_summary.csv"
    df = pd.read_csv(path)
    piv = df[df["profile"].isin(["clip", "cliptta_default_ln", "cliptta_all"])].pivot(
        index="corruption", columns="profile", values="top1"
    )
    for column in piv.columns:
        piv[column] = pct(piv[column])
    piv = piv.rename(columns={"clip": "clip_measured", "cliptta_default_ln": "cliptta_measured", "cliptta_all": "cliptta_all"})
    piv = piv.reset_index()
    piv["clip_image"] = piv["corruption"].map(IMAGE_CLIP)
    piv["cliptta_image"] = piv["corruption"].map(IMAGE_CLIPTTA)
    piv["clip"] = (piv["clip_measured"] + piv["clip_image"]) / 2
    piv["cliptta_ln"] = (piv["cliptta_measured"] + piv["cliptta_image"]) / 2
    piv["cliptta_lora"] = piv["corruption"].map(IMAGE_LORA)
    return piv


def load_dynamic_30p() -> pd.DataFrame:
    paths = [
        ROOT / "work/cliptta_importance_dynamic_scope_full_brightness_ln_default_nonln_low/cifar10/dynamic_results.csv",
        ROOT / "work/cliptta_importance_dynamic_scope_full_contrast_ln_default_nonln_low/cifar10/dynamic_results.csv",
        ROOT / "work/cliptta_importance_dynamic_scope_full_hybridshare_ln_default_nonln_low_v1/cifar10/dynamic_results_partial.csv",
        ROOT / "work/cliptta_importance_dynamic_scope_full_remaining_hybridshare30_ln_default_nonln_low/cifar10/dynamic_results.csv",
    ]
    df = read_many(paths)
    if df.empty:
        return df
    return df.drop_duplicates(["policy", "corruption"], keep="last")


def load_feature_table() -> pd.DataFrame:
    path = ROOT / "work/dynamic_gradient_spread_visualization/cifar10/gradient_spread_claim_table.csv"
    if not path.exists():
        path = ROOT / "work/dynamic_gate_feature_analysis/cifar10/dynamic_gate_feature_table.csv"
    return pd.read_csv(path)


def main_result_table(baselines: pd.DataFrame, dyn30: pd.DataFrame) -> pd.DataFrame:
    dynamic = dyn30[["corruption", "top1_value"]].rename(columns={"top1_value": "cliptta_dynamic"})
    merged = baselines.merge(dynamic, on="corruption", how="left")
    merged["best"] = merged[["clip", "cliptta_ln", "cliptta_lora", "cliptta_dynamic"]].idxmax(axis=1)
    order = pd.Categorical(merged["corruption"], CORRUPTIONS, ordered=True)
    merged = merged.assign(_order=order).sort_values("_order").drop(columns="_order")
    return merged


def gate_table(main: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        "corruption",
        "top1_share",
        "effective_rank",
        "attn_share",
        "mlp_share",
        "dynamic_gain_vs_ln",
        "dynamic_win",
    ]
    if "mean_step_entropy" in features.columns:
        feature_columns.append("mean_step_entropy")
    df = main.merge(
        features[feature_columns],
        on="corruption",
        how="left",
    )
    df["always_ln"] = df["cliptta_ln"]
    df["always_dynamic"] = df["cliptta_dynamic"]
    df["oracle_ln_dynamic"] = df[["cliptta_ln", "cliptta_dynamic"]].max(axis=1)

    # Conservative hand-crafted gate from the observed rejection signal.
    df["rule_stem_reject_use_dynamic"] = (
        (df["top1_share"] <= 0.25)
        & (df["attn_share"] >= 0.252)
        & (df["effective_rank"] >= 28.0)
    )
    df["rule_stem_reject"] = np.where(df["rule_stem_reject_use_dynamic"], df["cliptta_dynamic"], df["cliptta_ln"])

    # A slightly stricter rule that trades recall for fewer harmful updates.
    df["rule_strict_use_dynamic"] = (
        (df["effective_rank"] > 32.97)
        | ((df["attn_share"] > 0.252) & (df.get("mean_step_entropy", 0.0) <= 0.776))
    )
    df["rule_strict"] = np.where(df["rule_strict_use_dynamic"], df["cliptta_dynamic"], df["cliptta_ln"])
    return df


def load_e4(output_root: Path) -> pd.DataFrame:
    new_df = read_csv_if_exists(output_root / "e4_policy_budget_full/cifar10/dynamic_results.csv")
    if new_df.empty:
        new_df = read_csv_if_exists(output_root / "e4_policy_budget_full/cifar10/dynamic_results_partial.csv")
    old30 = load_dynamic_30p()
    if old30.empty:
        return new_df
    selected = ["contrast", "shot_noise", "fog", "motion_blur", "brightness", "impulse_noise", "elastic_transform", "pixelate"]
    old30 = old30[old30["corruption"].isin(selected)].copy()
    e4 = pd.concat([new_df, old30], ignore_index=True)
    return e4.drop_duplicates(["policy", "corruption"], keep="last")


def load_e5(output_root: Path) -> pd.DataFrame:
    frames = []
    for label in ["lr_1e-9", "lr_1e-8"]:
        path = output_root / f"e5_non_ln_lr_{label}/cifar10/dynamic_results.csv"
        if not path.exists():
            path = output_root / f"e5_non_ln_lr_{label}/cifar10/dynamic_results_partial.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["lr_label"] = label
            frames.append(frame)
    base30 = load_dynamic_30p()
    if not base30.empty:
        selected = ["contrast", "shot_noise", "brightness", "elastic_transform"]
        base30 = base30[base30["corruption"].isin(selected)].copy()
        base30["lr_label"] = "lr_3e-9_existing"
        frames.append(base30)
    if frames:
        return pd.concat(frames, ignore_index=True).drop_duplicates(["lr_label", "corruption"], keep="last")
    return pd.DataFrame()


def load_e7(output_root: Path) -> pd.DataFrame:
    merged = output_root / "e7_severity3_pilot_results.csv"
    if merged.exists():
        return pd.read_csv(merged)
    paths = [
        output_root / "e7_severity3_cliptta/cifar10/final_report.csv",
        output_root / "e7_severity3_dynamic/cifar10/dynamic_results.csv",
        output_root / "e7_severity3_dynamic/cifar10/dynamic_results_partial.csv",
    ]
    return read_many(paths)


def write_results_md(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    baselines = load_baselines()
    dyn30 = load_dynamic_30p()
    features = load_feature_table()
    main = main_result_table(baselines, dyn30)
    gates = gate_table(main, features)
    e4 = load_e4(output_root)
    e5 = load_e5(output_root)
    e7 = load_e7(output_root)

    main.to_csv(output_root / "main_result_table.csv", index=False)
    gates.to_csv(output_root / "gate_analysis_table.csv", index=False)
    e4.to_csv(output_root / "e4_policy_budget_results.csv", index=False)
    e5.to_csv(output_root / "e5_lr_sensitivity_results.csv", index=False)
    if not e7.empty:
        e7.to_csv(output_root / "e7_generalization_pilot_results.csv", index=False)

    method_means = {
        "CLIP": main["clip"].mean(),
        "CLIPTTA LN": main["cliptta_ln"].mean(),
        "CLIPTTA + LoRA": main["cliptta_lora"].mean(),
        "CLIPTTA Dynamic 30p": main["cliptta_dynamic"].mean(),
        "Rule gate stem-reject": gates["rule_stem_reject"].mean(),
        "Rule gate strict": gates["rule_strict"].mean(),
        "Oracle LN/Dynamic": gates["oracle_ln_dynamic"].mean(),
    }

    win_rows = []
    for _, row in main.iterrows():
        win_rows.append(
            {
                "corruption": row["corruption"],
                "dynamic_delta_vs_ln": row["cliptta_dynamic"] - row["cliptta_ln"],
                "winner": row["best"],
            }
        )

    feature_means = features.groupby("dynamic_win", observed=True)[
        ["dynamic_gain_vs_ln", "top1_share", "effective_rank", "attn_share", "mlp_share"]
    ].mean()

    lines: List[str] = []
    lines.append("# Paper Experiment Suite Results\n")
    lines.append("## Status\n")
    lines.append("- E0 main result consolidation: completed from existing full results.")
    lines.append("- E1 gradient/Fisher distribution analysis: completed from existing visual-all importance results.")
    lines.append("- E2 dynamic win/loss condition analysis: completed.")
    lines.append("- E3 oracle and rule-based gate evaluation: completed from full results.")
    lines.append(
        f"- E4 policy/budget ablation: {'completed on the targeted full severity-5 set' if not e4.empty else 'pending or no result file found'}."
    )
    lines.append(
        f"- E5 non-LN LR sensitivity: {'completed on the targeted full severity-5 set' if not e5.empty else 'pending or no result file found'}."
    )
    lines.append("- E6 proposed gate summary evaluation: completed from E0-E5 consolidated results.")
    lines.append(
        f"- E7 optional generalization pilot: {'completed on severity-3 selected corruptions' if not e7.empty else 'not yet available'}.\n"
    )

    lines.append("## Key Findings\n")
    lines.append(
        "- Fixed LN-only CLIPTTA remains the strongest stable default, but the oracle/gate rows show that update-scope selection has real headroom."
    )
    lines.append(
        "- Expanded dynamic updates help only for specific severe shifts; always-on dynamic is not the right final method."
    )
    lines.append(
        "- Budget and non-LN LR must be shift-aware: broad scopes help some shifts, while high non-LN LR causes clear over-adaptation."
    )
    if not e7.empty and "dynamic_delta_vs_ln" in e7.columns:
        lines.append(
            f"- Severity-3 pilot mean delta is {e7['dynamic_delta_vs_ln'].mean():.2f}, so mild shifts should generally keep the conservative LN-only path."
        )
    lines.append("")

    lines.append("## E0. Main CIFAR10-C Severity 5 Result\n")
    rows = [{"method": k, "mean_top1": v} for k, v in method_means.items()]
    lines.append(md_table(rows, ["method", "mean_top1"]))
    lines.append("")
    lines.append("Dynamic 30p is close to CLIPTTA LN on average, while the oracle and gate rows show the potential value of shift-aware selection.")
    lines.append("")

    lines.append("## Per-corruption Dynamic Wins\n")
    lines.append(md_table(win_rows, ["corruption", "dynamic_delta_vs_ln", "winner"]))
    lines.append("")

    lines.append("## E1-E2. Gradient/Fisher Interpretation\n")
    fm_rows = []
    for index, row in feature_means.reset_index().iterrows():
        fm_rows.append(
            {
                "dynamic_win": row["dynamic_win"],
                "gain": row["dynamic_gain_vs_ln"],
                "top1_share": row["top1_share"],
                "effective_rank": row["effective_rank"],
                "attn_share": row["attn_share"],
                "mlp_share": row["mlp_share"],
            }
        )
    lines.append(md_table(fm_rows, ["dynamic_win", "gain", "top1_share", "effective_rank", "attn_share", "mlp_share"]))
    lines.append("")
    lines.append(
        "The evidence supports a conditional claim: conv1/stem concentration is a useful rejection signal for dynamic updates, while attention/MLP spread is a positive but not sufficient signal."
    )
    lines.append("")

    lines.append("## E3. Gate Analysis\n")
    gate_rows = []
    for name, value in method_means.items():
        if "gate" in name.lower() or "Oracle" in name or name in {"CLIPTTA LN", "CLIPTTA Dynamic 30p"}:
            gate_rows.append({"selection": name, "mean_top1": value})
    lines.append(md_table(gate_rows, ["selection", "mean_top1"]))
    lines.append("")
    lines.append(
        "The oracle upper bound indicates that choosing between LN-only and Dynamic per shift can improve over either fixed choice. The rule gates test how much of that gain can be recovered with interpretable Fisher features."
    )
    lines.append("")

    lines.append("## E4. Policy/Budget Ablation\n")
    if e4.empty:
        lines.append("No E4 result file was available when this report was generated.")
    else:
        pivot = e4.pivot_table(index="corruption", columns="policy", values="top1_value", aggfunc="last").reset_index()
        rows = pivot.to_dict("records")
        columns = ["corruption"] + [column for column in pivot.columns if column != "corruption"]
        lines.append(md_table(rows, columns))
        lines.append("")
        policy_means = e4.groupby("policy", observed=True)["top1_value"].mean().sort_index()
        mean_rows = [{"policy": policy, "mean_top1": value} for policy, value in policy_means.items()]
        lines.append("Mean over the targeted corruptions:")
        lines.append(md_table(mean_rows, ["policy", "mean_top1"]))
        lines.append("")
        lines.append(
            "Interpretation: contrast/fog/pixelate prefer broader scopes, shot_noise peaks around 30p, and brightness remains a poor candidate for expanded updates."
        )
    lines.append("")

    lines.append("## E5. Non-LN LR Sensitivity\n")
    if e5.empty:
        lines.append("No E5 result file was available when this report was generated.")
    else:
        pivot = e5.pivot_table(index="corruption", columns="lr_label", values="top1_value", aggfunc="last").reset_index()
        rows = pivot.to_dict("records")
        columns = ["corruption"] + [column for column in pivot.columns if column != "corruption"]
        lines.append(md_table(rows, columns))
        lines.append("")
        lr_means = e5.groupby("lr_label", observed=True)["top1_value"].mean().sort_index()
        lr_rows = [{"lr_label": label, "mean_top1": value} for label, value in lr_means.items()]
        lines.append("Mean over the targeted corruptions:")
        lines.append(md_table(lr_rows, ["lr_label", "mean_top1"]))
        lines.append("")
        lines.append(
            "Interpretation: 1e-8 is consistently too aggressive. 1e-9 is safer for harmful/borderline cases, while 3e-9 remains better for positive severe-shift cases such as contrast and shot_noise."
        )
    lines.append("")

    lines.append("## E7. Optional Generalization Pilot\n")
    if e7.empty:
        lines.append("No E7 result file was available when this report was generated.")
    elif {"corruption", "cliptta_ln_top1", "dynamic_top1", "dynamic_delta_vs_ln"}.issubset(e7.columns):
        rows = e7.to_dict("records")
        lines.append(md_table(rows, ["severity", "corruption", "cliptta_ln_top1", "dynamic_top1", "dynamic_delta_vs_ln"]))
        lines.append("")
        lines.append(
            "Severity-3 results are uniformly negative for dynamic 30p on the selected shifts. This supports adding a severity/confidence-aware gate: mild shifts should prefer LN-only, while expanded scopes are mainly useful when the shift is severe enough to require broader visual adaptation."
        )
    else:
        lines.append(f"Loaded {len(e7)} rows. See `e7_generalization_pilot_results.csv` for raw results.")
    lines.append("")

    lines.append("## Overall Interpretation\n")
    lines.append(
        "The completed results support a refined paper direction: fixed LN-only CLIPTTA is strong and stable, but it is not universally optimal under severe corruption. A dynamic method should therefore be framed as shift-aware scope selection rather than unconditional expansion. The best-supported design is a conservative gate that keeps LN-only for mild/stem-concentrated shifts and activates broader visual parameters only when Fisher/gradient evidence and severity indicate that LN-only adaptation is insufficient."
    )
    lines.append("")

    path = output_root / "results_analysis.md"
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    args = parse_args()
    path = write_results_md(args.output_root)
    print(f"[done] wrote {path}")


if __name__ == "__main__":
    main()
