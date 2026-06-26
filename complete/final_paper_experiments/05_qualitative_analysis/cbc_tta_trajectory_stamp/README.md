# CBC + TTA Qualitative Trajectory

- `cbc_logit_shift_3d_*.png`: 3D class-logit evidence shift from CLIP to CBC.
- `tta_logit_trajectory_3d_*.png`: step-wise 3D class-evidence paths toward ideal class evidence.
- `tta_feature_trajectory_3d_*.png`: step-wise 3D feature-centroid paths relative to clean feature centroids.
- `tta_feature_distribution_steps_3d_*.png`: selected-step 3D sample distribution restoration panels.
- `feature_distribution_3d_frames/*.png`: every-step 3D sample distribution frames.
- `tta_stepwise_metrics_*.png`: step-wise top-1, centroid-text alignment, and inter/intra cluster ratio.
- CBC is used only to form pseudo labels; feature movement is produced by visual encoder adaptation.
- The adapter updates CLIP visual LayerNorm parameters to isolate pseudo-label guidance while keeping the experiment stable.
