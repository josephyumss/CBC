# TTA Before/After Feature PCA-3D

This folder visualizes feature distributions before and after adaptation for the non-CLIPTTA TTA methods used in the paper experiments.

- Circles: corrupted features before adaptation.
- Triangles: features after adaptation.
- Diamonds: class centroids after adaptation.
- Stars: default-prompt text prototypes.
- Arrows: class-centroid movement from before to after adaptation.
- TDA and ETTA are cache-based methods; they do not update the CLIP visual encoder, so their feature distributions are expected to overlap before/after.
