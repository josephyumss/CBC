# Offline CBC Global Centroid Bias Visualization

This folder contains qualitative visualizations for brightness and shot_noise.

- `*_global_centroid_text_pca3d.png`: corrupted global feature centroid, CBC-driven centroid, and the text centroid in one PCA-3D space.
- `*_centroid_text_probabilities_bar.png`: softmax probability distribution over text prototypes before/after CBC.
- Offline CBC estimates one class-logit bias using all test samples from the corruption and subtracts it from every sample.
- After CBC, the global centroid has nearly equal logits against all text prototypes; after softmax this appears as an almost uniform green probability bar around 0.1.
