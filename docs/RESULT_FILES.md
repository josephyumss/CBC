# Result File Map

The canonical final results are in `complete/final_paper_experiments/`.

## Paper Sections

- `01_main_result_corruption_s5/`: corruption-wise severity-5 zero-shot comparison.
- `02_robustness_online/`: corruption mixed, severity mixed, class imbalance, and batch-size sweep.
- `03_offline_setting/`: CLIP, CBC Offline, InMaP, Frolic, and additional offline baselines.
- `04_clean_image/`: clean CIFAR-10 CLIP/CBC and zero-shot baselines.
- `05_qualitative_analysis/`: PCA, centroid, probability, and distribution-shape visualizations.
- `06_cbc_with_tta_methods/`: 9 TTA methods before/after CBC.

## Main Paper Figures

- `complete/final_paper_experiments/paper_figures/batch_size_sweep_online.png`
- `complete/final_paper_experiments/paper_figures/class_imbalance_sweep_online.png`
- `complete/final_paper_experiments/05_qualitative_analysis/cbc_global_centroid_bias/shot_noise/shot_noise_global_centroid_text_pca3d.png`
- `complete/final_paper_experiments/05_qualitative_analysis/cbc_global_centroid_bias/shot_noise/shot_noise_centroid_text_probabilities_bar.png`

## Main CSV Tables

- `complete/final_paper_experiments/01_main_result_corruption_s5/main_corruption_s5_overall.csv`
- `complete/final_paper_experiments/02_robustness_online/severity_mixed_s1_s5.csv`
- `complete/final_paper_experiments/02_robustness_online/class_imbalance_average_online.csv`
- `complete/final_paper_experiments/02_robustness_online/batch_size_sweep_online.csv`
- `complete/final_paper_experiments/03_offline_setting/offline_setting_overall_s5.csv`
- `complete/final_paper_experiments/04_clean_image/clean_online_offline.csv`
- `complete/final_paper_experiments/06_cbc_with_tta_methods/tta_overall_summary_19methods.csv`
