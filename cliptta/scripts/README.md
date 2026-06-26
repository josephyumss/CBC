# Curated Experiment Scripts

This folder keeps only the scripts needed to reproduce or regenerate the final
CBC paper experiments.

## Core Evaluation

- `run_zero_shot_enhancer_full_experiment_suite.py`: CLIP, CBC, CALIP, WaffleCLIP, CuPL, InMaP, Frolic, severity sweep, batch sweep, and zero-shot tables.
- `evaluate_zero_shot_enhancement_baselines.py`: shared baseline implementations and CLIP/CBC evaluation helpers.
- `evaluate_additional_zero_shot_baselines_from_cache.py`: additional online/offline baseline screening from cached logits.
- `run_closed_set_cifar_benchmarks.py`: common CIFAR/CIFAR-C dataset/model utilities.
- `run_cbc_clip_zero_shot_experiments.py`: CLIP/CBC zero-shot experiment driver.
- `run_clip_cbc_clean_corrupted_eval.py`: clean/corrupted CLIP/CBC evaluation.
- `run_cbc_class_imbalance_sweep.py`: class imbalance sweep from cached per-sample logits.
- `run_pseudolabel_calibration_baseline_suite.py`: TTA raw vs TTA+CBC evaluation.
- `profile_zero_shot_enhancement_efficiency.py`: FLOPs, memory, and throughput profiling.

## Analysis and Figure Generation

- `analyze_paper_experiment_suite.py`: final table aggregation.
- `plot_feature_cluster_pca_3d.py`: per-corruption PCA feature clusters.
- `plot_clean_corruption_distribution_similarity.py`: clean/corrupted distribution-shape similarity.
- `plot_feature_center_vs_shape_degradation.py`: center-shift vs geometry-distortion analysis.
- `plot_cbc_global_centroid_bias_qualitative.py`: brightness/shot-noise centroid and probability figures.
- `plot_cbc_class_centroid_shift_pca_3d.py`: class centroid shift PCA.
- `plot_tta_before_after_feature_pca_3d.py`: TTA before/after feature PCA.
- `plot_cbc_tta_trajectory_qualitative.py`: CBC/TTA trajectory visualizations.
