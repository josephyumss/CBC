# Feature Center Shift vs Shape Distortion

This experiment starts from clean CLIP image features and injects synthetic perturbations along separated axes.

Recommended figure for the paper:

- `controlled_matched_feature_l2_budget_degradation.png`

Interpretation:

- Center shift applies a shared CBC-aligned feature displacement to every clean image feature.
- Geometry Distortion is implemented as a class-centered, within-class permuted residual, preserving class centers while changing feature geometry.
- The paired residual curve in `matched_feature_l2_budget_degradation.png` should be treated as a diagnostic, not as the main evidence, because the paired clean-to-corrupted residual can contain sample-specific semantic directions that occasionally improve clean CLIP accuracy.
- The controlled plot shows only Clean CLIP, Center Shift, and Geometry Distortion under the same mean feature L2 budget.
- Under the same feature perturbation budget, CLIP predictions are much more sensitive to center displacement than to center-preserving shape changes, which supports CBC as a lightweight correction for corruption-induced center/bias shift.

The full empirical corruption residual has a larger total norm than the mean shift, so full corruption accuracy drop is not explained by center shift alone. This analysis is meant to isolate per-unit sensitivity, not to claim that all corruption damage comes only from center shift.
