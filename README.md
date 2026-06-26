# Corruption Bias Calibration (CBC) Release

This directory is a cleaned release bundle for the CBC project.  It keeps the
final experiment code, final result tables/figures, and only the core experiment
logs needed to understand the paper results.

## Directory Layout

```text
cbc_github_release/
├── README.md
├── TimesNewerRoman.zip
├── cliptta/
│   ├── ttavlm/                         # CLIP/TTA implementation used by experiments
│   ├── scripts/                        # Curated experiment/reproduction scripts
│   ├── requirements.txt
│   └── setup.py
├── complete/final_paper_experiments/   # Final paper-ready CSVs, figures, and LaTeX tables
├── external_zero_shot_baselines/        # Local copies of zero-shot baseline code
├── ETTA/                               # ETTA code only; large DATA/ and .git/ omitted
└── TDA/                                # TDA code only; .git/ omitted
```

Raw caches, intermediate failed runs, checkpoints, `cliptta/work/`, and bulky
runtime logs are intentionally omitted.

## Core Experimental Setting

- Model: CLIP ViT-B/16.
- Dataset: CIFAR-10 and CIFAR-10-C.
- Main corruption setting: CIFAR-10-C severity 5, 15 corruptions.
- Prompt: default template only, `a photo of a {}`.
- Online CBC batch size: 128 unless a sweep explicitly changes it.
- Seed: 42.

CBC applies class-wise logit centering:

```text
logits_cbc = logits - mean_batch(logits)
```

For offline CBC, the same bias is estimated once from the full evaluation set.

## Final Core Logs

### 1. Main Result: CIFAR-10-C Severity 5

Main result is reported as the corruption-wise macro average: each of the 15
corruptions is evaluated independently, and the final score is averaged across
corruption types.

| Method | Top-1 (%) | Δ vs. CLIP |
|---|---:|---:|
| CLIP | 60.19 | +0.00 |
| CBC | 66.73 | +6.53 |
| CALIP | 57.40 | -2.79 |
| WaffleCLIP | 63.20 | +3.01 |
| CuPL | 63.08 | +2.89 |

Source: `complete/final_paper_experiments/01_main_result_corruption_s5/main_corruption_s5_long.csv`

### 2. Robustness

| Setting | CLIP | CBC | Δ |
|---|---:|---:|---:|
| Corruption mixed, severity 5 | 60.19 | 64.73 | +4.54 |
| Severity mixed, severity 1--5 | 72.77 | 76.46 | +3.69 |
| Clean CIFAR-10 | 89.25 | 91.16 | +1.91 |

Additional robustness logs:

- Class imbalance sweep: CBC remains above CLIP up to 86% imbalance.
- Batch-size sweep: streaming CBC is 66.83%; batch 128 CBC is 66.73%; batch 512 CBC is 66.80%.

Sources:

- `complete/final_paper_experiments/02_robustness_online/corruption_mixed_s5.csv`
- `complete/final_paper_experiments/02_robustness_online/severity_mixed_s1_s5.csv`
- `complete/final_paper_experiments/02_robustness_online/class_imbalance_average_online.csv`
- `complete/final_paper_experiments/02_robustness_online/batch_size_sweep_online.csv`
- `complete/final_paper_experiments/04_clean_image/clean_online_offline.csv`

### 3. Offline Setting

| Method | Top-1 (%) | Δ vs. CLIP |
|---|---:|---:|
| CLIP | 60.19 | +0.00 |
| CBC Offline | 64.98 | +4.79 |
| InMaP | 67.23 | +7.03 |
| Frolic | 66.76 | +6.57 |

Source: `complete/final_paper_experiments/03_offline_setting/offline_setting_overall_s5.csv`

### 4. CBC with TTA Methods

| TTA Method | Raw Top-1 | +CBC Top-1 | Δ |
|---|---:|---:|---:|
| WATT | 52.75 | 74.93 | +22.18 |
| WATT-UnIEnt | 64.84 | 75.11 | +10.28 |
| STAMP | 60.17 | 66.70 | +6.53 |
| AdaContrast | 59.68 | 66.05 | +6.37 |
| WATT-OTSU | 65.12 | 69.83 | +4.71 |
| ETTA | 62.27 | 66.98 | +4.70 |
| TDA | 61.84 | 66.46 | +4.61 |
| ClipartT | 68.04 | 71.40 | +3.36 |
| UnIEnt | 79.28 | 75.98 | -3.30 |

Source: `complete/final_paper_experiments/06_cbc_with_tta_methods/tta_overall_summary_19methods.csv`

### 5. Qualitative Shot-Noise Check

CBC removes the shared centroid-text bias almost completely in the shot-noise
example used in the paper figure.

| Metric | Before CBC | After CBC |
|---|---:|---:|
| Centroid-text logit std | 8.47e-3 | 2.44e-8 |
| Centroid-text logit range | 2.42e-2 | 7.21e-8 |
| Centroid-text probability std | 8.48e-4 | 1.03e-8 |
| Centroid-text probability range | 2.43e-3 | 7.45e-9 |

Source: `complete/final_paper_experiments/05_qualitative_analysis/cbc_global_centroid_bias/summary.csv`

Main qualitative files:

- `complete/final_paper_experiments/05_qualitative_analysis/cbc_global_centroid_bias/shot_noise/shot_noise_global_centroid_text_pca3d.png`
- `complete/final_paper_experiments/05_qualitative_analysis/cbc_global_centroid_bias/shot_noise/shot_noise_centroid_text_probabilities_bar.png`
- `complete/final_paper_experiments/paper_figures/batch_size_sweep_online.png`
- `complete/final_paper_experiments/paper_figures/class_imbalance_sweep_online.png`

## Reproduction

Create an environment, install dependencies, and run from the `cliptta/` folder:

```bash
cd cbc_github_release/cliptta
pip install -r requirements.txt
pip install -e .
```

Set your CIFAR/CIFAR-C root:

```bash
DATA_ROOT=/path/to/data
```

Run the zero-shot suite:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_zero_shot_enhancer_full_experiment_suite.py \
  --family cifar10 \
  --severity 5 \
  --data-root "$DATA_ROOT" \
  --output-root work/zero_shot_enhancement_full_suite \
  --run-name cifar10_full_zero_shot_enhancers \
  --batch-size 128 \
  --feature-batch-size 256 \
  --workers 2 \
  --seed 42
```

Run the TTA + CBC suite:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_pseudolabel_calibration_baseline_suite.py \
  --family cifar10 \
  --severity 5 \
  --shift-types all \
  --methods clipartt unient stamp tda adacontrast watt watt_otsu watt_unient etta \
  --variants raw calibrated \
  --calibration-prompt-mode default \
  --data-root "$DATA_ROOT" \
  --output-root work/cbc_tta_suite \
  --run-name cifar10c_s5_tta_cbc \
  --workers 2 \
  --seeds 42 \
  --steps 10 \
  --default-batch-size 128 \
  --display-progress
```

Regenerate the qualitative shot-noise/brightness centroid figures:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/plot_cbc_global_centroid_bias_qualitative.py \
  --data-root "$DATA_ROOT" \
  --device cuda:0 \
  --batch-size 256 \
  --num-workers 2
```

## Notes

- `complete/final_paper_experiments/` is the canonical final result folder.
- `cliptta/work/` is not tracked in this release; reruns will recreate it.
- External code folders are included only for experiment reproduction and should keep their original licenses if this bundle is published.
