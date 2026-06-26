#!/usr/bin/env python3
"""Run closed-set CIFAR benchmarks for CLIPTTA.

This script is designed around the CLIPTTA paper's closed-set CIFAR setup
using the CLIP ViT-B/16 backbone and the dataset layout prepared by
`scripts/prepare_cifar_closed_set_data.py`.

Two benchmark suites are provided:

- `paper`: implemented baselines from the paper's coarse-grained/closed-set
  CIFAR comparison, excluding TPT which is not implemented in this repo.
- `all_repo`: the paper suite plus additional closed-set-capable methods that
  exist in this repository. These extra methods are convenient for exploration,
  but they are not part of the paper's main CIFAR comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/home/josephyumss/data")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "work" / "closed_set_cifar_benchmarks"
UNIMPLEMENTED_PAPER_BASELINES = ["tpt"]
UNSUPPORTED_REPO_ADAPTATIONS = [
    "norm",  # Listed in configuration choices but not implemented in ttavlm.methods.return_tta_model.
    "tent_oracle",  # Requires oracle labels and is not a standard closed-set test-time baseline.
]


FAMILIES = OrderedDict(
    {
        "cifar10": {
            "clean_dataset": "cifar10",
            "corrupted_dataset": "cifar10c",
            "num_classes": 10,
            "cliptta_num_shots": 4,
            "cliptta_sample_size": 40,
        },
        "cifar100": {
            "clean_dataset": "cifar100",
            "corrupted_dataset": "cifar100c",
            "num_classes": 100,
            "cliptta_num_shots": 16,
            "cliptta_sample_size": 128,
        },
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=["paper", "all_repo"],
        default="paper",
        help="Which benchmark suite to run.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=list(FAMILIES.keys()),
        default=list(FAMILIES.keys()),
        help="Which CIFAR families to run.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Dataset root prepared for CLIPTTA.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Where logs and per-run metadata should be stored.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Dataloader worker count.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Seeds passed to ttavlm.main.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to launch ttavlm.main.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size forwarded to ttavlm.main for batch-compatible baselines.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES override for child runs.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable the repo's DataParallel flag.",
    )
    parser.add_argument(
        "--display-progress",
        action="store_true",
        help="Forward --display_progress to ttavlm.main.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep going when one baseline fails.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs that already have a successful summary.json.",
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Run only CIFAR-10/CIFAR-100 clean evaluations.",
    )
    parser.add_argument(
        "--corruptions-only",
        action="store_true",
        help="Run only CIFAR-10-C/CIFAR-100-C evaluations.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Optional subset of baseline keys to run.",
    )
    return parser.parse_args()


def bool_flag(flag: str, enabled: bool) -> List[str]:
    return [flag] if enabled else []


def stdout_only_enabled() -> bool:
    return os.getenv("TOSTTA_STDOUT_ONLY", "").lower() in {"1", "true", "yes"}


def parse_prettytable_rows(output: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells[0] == "dataset":
            continue
        if len(cells) == 5:
            rows.append(
                {
                    "dataset": cells[0],
                    "corruption": cells[1],
                    "acc": cells[2],
                    "top1": cells[2],
                    "top5": "",
                    "auroc": cells[3],
                    "fpr95": cells[4],
                }
            )
            continue
        if len(cells) != 7:
            continue
        rows.append(
            {
                "dataset": cells[0],
                "corruption": cells[1],
                "acc": cells[2],
                "top1": cells[3],
                "top5": cells[4],
                "auroc": cells[5],
                "fpr95": cells[6],
            }
        )
    return rows


def load_success(summary_path: Path) -> bool:
    if not summary_path.exists():
        return False
    with summary_path.open() as handle:
        summary = json.load(handle)
    return summary.get("returncode") == 0


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def common_flags(
    args: argparse.Namespace,
    dataset_name: str,
    exp_name: str,
    save_root: Path,
    batch_size: int,
) -> List[str]:
    flags = [
        "--env",
        "closed_set_cifar",
        "--exp_name",
        exp_name,
        "--dataroot",
        str(args.data_root),
        "--save_root",
        str(save_root),
        "--dataset",
        dataset_name,
        "--base_model_name",
        "clip-ViT-B/16",
        "--steps",
        "10",
        "--workers",
        str(args.workers),
        "--batch_size",
        str(batch_size),
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--closed_set",
    ]

    if dataset_name.endswith("c"):
        flags += ["--shift_type", "all"]
    else:
        flags += ["--shift_type", "original"]

    flags += bool_flag("--distributed", args.distributed)
    flags += bool_flag("--display_progress", args.display_progress)
    return flags


def baseline_clip(_: str, __: Dict[str, object], ___: bool) -> List[str]:
    return ["--adaptation", "source"]


def baseline_tent(_: str, __: Dict[str, object], ___: bool) -> List[str]:
    return [
        "--adaptation",
        "tent",
        "--score_type",
        "max_prob",
        "--beta_tta",
        "1.0",
        "--beta_reg",
        "0.0",
    ]


def baseline_eta(_: str, __: Dict[str, object], ___: bool) -> List[str]:
    return [
        "--adaptation",
        "eta",
        "--score_type",
        "max_prob",
        "--beta_tta",
        "1.0",
        "--beta_reg",
        "0.0",
        "--alpha_entropy",
        "0.4",
        "--d_margin",
        "0.05",
    ]


def baseline_sar(_: str, __: Dict[str, object], ___: bool) -> List[str]:
    return [
        "--adaptation",
        "sar",
        "--score_type",
        "max_prob",
        "--beta_tta",
        "1.0",
        "--beta_reg",
        "0.0",
        "--alpha_entropy",
        "0.4",
        "--reset_constant_em",
        "0.2",
        "--use_sam",
    ]


def baseline_rotta(_: str, family_cfg: Dict[str, object], __: bool) -> List[str]:
    batch_size = 128
    return [
        "--adaptation",
        "rotta",
        "--capacity",
        str(batch_size),
        "--update_frequency",
        str(batch_size),
        "--lambda_u",
        "1.0",
        "--lambda_t",
        "1.0",
        "--alpha_rotta",
        "0.05",
        "--nu",
        "0.001",
        "--use_tta",
        "--batch_size",
        str(batch_size),
    ]


def baseline_clipartt(_: str, __: Dict[str, object], ___: bool) -> List[str]:
    return [
        "--adaptation",
        "clipartt",
        "--lr",
        "1e-4",
        "--K",
        "3",
    ]


def baseline_watt(_: str, __: Dict[str, object], ___: bool) -> List[str]:
    return [
        "--adaptation",
        "watt",
        "--batch_size",
        "128",
        "--template_type",
        "select",
        "--lr",
        "1e-4",
        "--avg_type",
        "sequential",
        "--meta_reps",
        "5",
        "--reps",
        "2",
    ]


def baseline_tda(_: str, __: Dict[str, object], ___: bool) -> List[str]:
    return [
        "--adaptation",
        "tda",
        "--batch_size",
        "1",
        "--pos_alpha_beta",
        "2.0",
        "2.0",
        "--neg_alpha_beta",
        "0.117",
        "1.0",
        "--entropy_threshold",
        "0.2",
        "0.5",
        "--mask_threshold",
        "0.03",
        "1.0",
        "--pos_shot_capacity",
        "2",
        "--neg_shot_capacity",
        "3",
    ]


def baseline_cliptta(_: str, family_cfg: Dict[str, object], __: bool) -> List[str]:
    return [
        "--adaptation",
        "cliptta",
        "--lr",
        "1e-4",
        "--beta_tta",
        "1.0",
        "--beta_reg",
        "1.0",
        "--id_score_type",
        "max_prob",
        "--use_softmax_entropy",
        "--use_memory",
        "--num_shots",
        str(family_cfg["cliptta_num_shots"]),
        "--sample_size",
        str(family_cfg["cliptta_sample_size"]),
    ]


def baseline_cliptta_fast(_: str, family_cfg: Dict[str, object], __: bool) -> List[str]:
    return [
        "--adaptation",
        "cliptta_fast",
        "--batch_size",
        "128",
        "--num_shots",
        str(family_cfg["cliptta_num_shots"]),
        "--sample_size",
        str(family_cfg["cliptta_sample_size"]),
        "--fast_cache_alpha",
        "0.2",
        "--fast_cache_beta",
        "5.0",
        "--fast_update_ratio",
        "0.1",
        "--fast_entropy_threshold",
        "1.0",
    ]


def baseline_cbc_contrastive(_: str, __: Dict[str, object], ___: bool) -> List[str]:
    return [
        "--adaptation",
        "cbc_contrastive",
        "--batch_size",
        "128",
        "--steps",
        "1",
        "--optimizer_type",
        "adam",
        "--lr",
        "1e-5",
        "--beta_tta",
        "1.0",
        "--beta_reg",
        "0.0",
        "--id_score_type",
        "max_prob",
        "--cbc_shift_weight",
        "1.0",
        "--cbc_contrastive_weight",
        "1.0",
        "--cbc_proto_weight",
        "1.0",
        "--cbc_diversity_weight",
        "0.2",
        "--cbc_target_mode",
        "classwise",
    ]


def baseline_lame(_: str, __: Dict[str, object], ___: bool) -> List[str]:
    return ["--adaptation", "lame"]


def baseline_ostta(_: str, __: Dict[str, object], ___: bool) -> List[str]:
    return ["--adaptation", "ostta", "--beta_tta", "1.0", "--beta_reg", "0.0"]


def baseline_sotta(_: str, family_cfg: Dict[str, object], __: bool) -> List[str]:
    threshold = 1.0 / int(family_cfg["num_classes"])
    return [
        "--adaptation",
        "sotta",
        "--capacity",
        "128",
        "--high_threshold",
        f"{threshold}",
        "--use_sam",
    ]


def baseline_adacontrast(_: str, __: Dict[str, object], ___: bool) -> List[str]:
    return [
        "--adaptation",
        "adacontrast",
        "--batch_size",
        "64",
        "--optimizer_type",
        "sgd",
        "--lr",
        "1e-3",
        "--beta_reg",
        "0.0",
    ]


def baseline_calip(_: str, __: Dict[str, object], ___: bool) -> List[str]:
    return ["--adaptation", "calip"]


def baseline_unient(_: str, __: Dict[str, object], ___: bool) -> List[str]:
    return [
        "--adaptation",
        "unient",
        "--beta_tta",
        "1.0",
        "--beta_reg",
        "1.0",
    ]


def baseline_zero(_: str, __: Dict[str, object], ___: bool) -> List[str]:
    return ["--adaptation", "zero", "--use_tta"]


def baseline_stamp(_: str, __: Dict[str, object], ___: bool) -> List[str]:
    return [
        "--adaptation",
        "stamp",
        "--batch_size",
        "64",
        "--optimizer_type",
        "sgd",
        "--lr",
        "1e-4",
        "--memory_length",
        "128",
        "--alpha_stamp",
        "0.45",
        "--use_consistency_filtering",
    ]


def baseline_cliptta_old(_: str, family_cfg: Dict[str, object], __: bool) -> List[str]:
    return [
        "--adaptation",
        "cliptta_old",
        "--lr",
        "1e-4",
        "--beta_tta",
        "1.0",
        "--beta_reg",
        "1.0",
        "--id_score_type",
        "max_prob",
        "--use_softmax_entropy",
        "--use_memory",
        "--num_shots",
        str(family_cfg["cliptta_num_shots"]),
        "--sample_size",
        str(family_cfg["cliptta_sample_size"]),
    ]


def baseline_watt_otsu(_: str, __: Dict[str, object], ___: bool) -> List[str]:
    return [
        "--adaptation",
        "watt_otsu",
        "--batch_size",
        "128",
        "--template_type",
        "select",
        "--lr",
        "1e-4",
        "--avg_type",
        "sequential",
        "--meta_reps",
        "5",
        "--reps",
        "2",
    ]


def baseline_watt_unient(_: str, __: Dict[str, object], ___: bool) -> List[str]:
    return [
        "--adaptation",
        "watt_unient",
        "--batch_size",
        "128",
        "--template_type",
        "select",
        "--lr",
        "1e-4",
        "--avg_type",
        "sequential",
        "--meta_reps",
        "5",
        "--reps",
        "2",
    ]


def baseline_etta(_: str, __: Dict[str, object], ___: bool) -> List[str]:
    return []


BASELINE_BUILDERS: "OrderedDict[str, Callable[[str, Dict[str, object], bool], List[str]]]" = OrderedDict(
    [
        ("clip", baseline_clip),
        ("tent", baseline_tent),
        ("eta", baseline_eta),
        ("sar", baseline_sar),
        ("rotta", baseline_rotta),
        ("clipartt", baseline_clipartt),
        ("watt", baseline_watt),
        ("tda", baseline_tda),
        ("cliptta", baseline_cliptta),
    ]
)

EXTRA_BUILDERS: "OrderedDict[str, Callable[[str, Dict[str, object], bool], List[str]]]" = OrderedDict(
    [
        ("lame", baseline_lame),
        ("ostta", baseline_ostta),
        ("sotta", baseline_sotta),
        ("adacontrast", baseline_adacontrast),
        ("calip", baseline_calip),
        ("unient", baseline_unient),
        ("stamp", baseline_stamp),
        ("zero", baseline_zero),
        ("etta", baseline_etta),
        ("cliptta_fast", baseline_cliptta_fast),
        ("cbc_contrastive", baseline_cbc_contrastive),
        ("cliptta_old", baseline_cliptta_old),
        ("watt_otsu", baseline_watt_otsu),
        ("watt_unient", baseline_watt_unient),
    ]
)


def get_suite_builders(suite: str) -> "OrderedDict[str, Callable[[str, Dict[str, object], bool], List[str]]]":
    if suite == "paper":
        return BASELINE_BUILDERS
    suite_builders = OrderedDict(BASELINE_BUILDERS)
    suite_builders.update(EXTRA_BUILDERS)
    return suite_builders


def iter_dataset_jobs(args: argparse.Namespace) -> Iterable[tuple[str, str, bool]]:
    for family in args.families:
        family_cfg = FAMILIES[family]
        if not args.corruptions_only:
            yield family, str(family_cfg["clean_dataset"]), False
        if not args.clean_only:
            yield family, str(family_cfg["corrupted_dataset"]), True


def parse_summary(rows: List[Dict[str, str]], dataset_name: str) -> Dict[str, object]:
    dataset_rows = [row for row in rows if row["dataset"] == dataset_name]
    overall_row = next((row for row in dataset_rows if row["corruption"] == "overall"), None)
    average_row = next((row for row in rows if row["dataset"] == "Average"), None)
    return {
        "overall": overall_row,
        "average": average_row,
        "rows": dataset_rows,
    }


def build_command(
    args: argparse.Namespace,
    builders: "OrderedDict[str, Callable[[str, Dict[str, object], bool], List[str]]]",
    method_name: str,
    family: str,
    family_cfg: Dict[str, object],
    dataset_name: str,
    is_corrupted: bool,
    run_dir: Path,
    save_root: Path,
) -> List[str]:
    if method_name == "etta":
        return [
            args.python,
            "scripts/run_closed_set_cifar_etta.py",
            "--dataset",
            dataset_name,
            "--data-root",
            str(args.data_root),
            "--workers",
            str(args.workers),
            "--seeds",
            *[str(seed) for seed in args.seeds],
        ]

    batch_size = 1 if method_name == "tda" else int(args.batch_size)
    exp_name = f"{method_name}_{dataset_name}_closed_set"
    return [
        args.python,
        "-m",
        "ttavlm.main",
        *common_flags(args, dataset_name, exp_name, save_root, batch_size),
        *builders[method_name](family, family_cfg, is_corrupted),
    ]


def main() -> None:
    args = parse_args()
    if args.clean_only and args.corruptions_only:
        raise SystemExit("Choose at most one of --clean-only and --corruptions-only.")

    builders = get_suite_builders(args.suite)
    if args.methods is not None:
        requested = []
        for method in args.methods:
            if method not in builders:
                available = ", ".join(builders.keys())
                raise SystemExit(f"Unknown method '{method}'. Available: {available}")
            requested.append(method)
        method_names = requested
    else:
        method_names = list(builders.keys())

    args.output_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "suite": args.suite,
        "families": args.families,
        "data_root": str(args.data_root),
        "output_root": str(args.output_root),
        "methods": method_names,
        "unimplemented_paper_baselines": UNIMPLEMENTED_PAPER_BASELINES if args.suite == "paper" else [],
        "unsupported_repo_adaptations": UNSUPPORTED_REPO_ADAPTATIONS,
    }
    write_text(args.output_root / "metadata.json", json.dumps(metadata, indent=2))
    mpl_config_dir = args.output_root / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []

    for family, dataset_name, is_corrupted in iter_dataset_jobs(args):
        family_cfg = FAMILIES[family]
        for method_name in method_names:
            run_dir = args.output_root / args.suite / method_name / dataset_name
            run_dir.mkdir(parents=True, exist_ok=True)
            summary_path = run_dir / "summary.json"
            if args.skip_existing and load_success(summary_path):
                print(f"[skip] {method_name} on {dataset_name}")
                continue

            save_root = run_dir / "artifacts"
            save_root.mkdir(parents=True, exist_ok=True)
            cmd = build_command(
                args=args,
                builders=builders,
                method_name=method_name,
                family=family,
                family_cfg=family_cfg,
                dataset_name=dataset_name,
                is_corrupted=is_corrupted,
                run_dir=run_dir,
                save_root=save_root,
            )

            env = os.environ.copy()
            if args.cuda_visible_devices is not None:
                env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
            env.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

            command_str = shlex.join(cmd)
            write_text(run_dir / "command.sh", command_str + "\n")
            print(f"[run] {method_name} on {dataset_name}")
            print(f"      {command_str}")

            if args.dry_run:
                log_path = "" if stdout_only_enabled() else str(run_dir / "stdout.log")
                summary_rows.append(
                    {
                        "suite": args.suite,
                        "method": method_name,
                        "dataset": dataset_name,
                        "status": "dry_run",
                        "log_path": log_path,
                    }
                )
                continue

            completed = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )
            combined_output_parts: List[str] = []
            assert completed.stdout is not None
            for line in completed.stdout:
                print(line, end="", flush=True)
                combined_output_parts.append(line)
            returncode = completed.wait()
            combined_output = "".join(combined_output_parts)
            log_path = "" if stdout_only_enabled() else str(run_dir / "stdout.log")
            if not stdout_only_enabled():
                write_text(run_dir / "stdout.log", combined_output)

            parsed_rows = parse_prettytable_rows(combined_output)
            parsed_summary = parse_summary(parsed_rows, dataset_name)
            summary = {
                "suite": args.suite,
                "method": method_name,
                "dataset": dataset_name,
                "family": family,
                "is_corrupted": is_corrupted,
                "returncode": returncode,
                "overall": parsed_summary["overall"],
                "average": parsed_summary["average"],
                "log_path": log_path,
                "save_root": str(save_root),
                "command": cmd,
            }
            write_text(summary_path, json.dumps(summary, indent=2))

            summary_rows.append(
                {
                    "suite": args.suite,
                    "method": method_name,
                    "dataset": dataset_name,
                    "status": "ok" if returncode == 0 else "failed",
                    "acc": (parsed_summary["overall"] or {}).get("acc", ""),
                    "top1": (parsed_summary["overall"] or {}).get("top1", ""),
                    "top5": (parsed_summary["overall"] or {}).get("top5", ""),
                    "auroc": (parsed_summary["overall"] or {}).get("auroc", ""),
                    "fpr95": (parsed_summary["overall"] or {}).get("fpr95", ""),
                    "log_path": log_path,
                }
            )

            if returncode != 0:
                print(f"[fail] {method_name} on {dataset_name} -> {returncode}")
                if not args.continue_on_error:
                    break
        else:
            continue
        break

    csv_path = args.output_root / "summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["suite", "method", "dataset", "status", "acc", "top1", "top5", "auroc", "fpr95", "log_path"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[done] wrote summary to {csv_path}")
    if args.suite == "paper":
        print(
            "[note] TPT is part of the paper's CIFAR comparison but is not implemented in this repository, "
            "so it is not included in the runnable suite."
        )


if __name__ == "__main__":
    main()
