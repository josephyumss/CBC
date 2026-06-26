# Clean vs Corrupted Feature Distribution Shape Similarity

This analysis compares CLIP image-feature distributions between clean CIFAR-10 and each CIFAR-10-C corruption at severity 5.

- Features are first CLIP-normalized by the model.
- To compare shape rather than location, we subtract each distribution mean and divide by its RMS radius.
- The main metric is covariance Frobenius cosine similarity after this center/scale normalization.
- Additional diagnostics compare within-class covariance and class-centroid geometry.

Use `global_distribution_shape_similarity.png` as the main corruption-wise visualization.
