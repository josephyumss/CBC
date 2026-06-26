# Qualitative Analysis

- Includes a 15-corruption sample grid and PCA visualizations for manually selected corruption types.
- Selected best-case corruption: `brightness`.
- Selected worst-case corruption: `glass_blur`.
- Files: `sample_15_corruptions_grid.png`, `pca_best_brightness/`, `pca_worst_glass_blur/`.
- Additional 3D trajectory analysis: `cbc_tta_trajectory_stamp/` visualizes CBC's logit-space shift, controlled STAMP-style TTA paths with/without CBC pseudo labels, and step-wise 3D feature-distribution restoration frames.
- CBC class-centroid shift analysis: `cbc_class_centroid_shift_3d/` shows where each corrupted class centroid moves under the CBC-equivalent feature displacement, with default-prompt text prototypes shown as class-wise stars.
- TTA before/after feature analysis: `tta_before_after_feature_pca3d/` compares pre- and post-adaptation feature distributions for the nine non-CLIPTTA TTA methods used in the paper.
- Center-vs-shape degradation analysis: `feature_center_vs_shape_degradation/` separates center shift from center-preserving shape changes; use `controlled_matched_feature_l2_budget_degradation.png` as the main figure.
- Distribution shape similarity analysis: `feature_distribution_similarity/` compares center/scale-normalized clean and corrupted CLIP feature distributions per corruption.
- Offline CBC global-centroid bias visualization: `cbc_global_centroid_bias/` shows brightness and shot_noise corrupted centroids, text prototypes, and CBC-shifted centroids with centroid-text logit bars.
