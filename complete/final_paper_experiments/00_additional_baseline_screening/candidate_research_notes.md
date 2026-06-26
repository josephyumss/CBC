# Additional Zero-shot Baseline Candidate Notes

## Online / non-transductive candidates considered

1. CLIP prompt ensemble / prompt logit ensemble
   - Source: OpenAI CLIP and common CLIP zero-shot evaluation practice.
   - Status here: evaluated from cached image features with select/all prompt sets.
   - Added files: `online_additional_overall_s5.csv`, `online_additional_clean.csv`.

2. WaffleCLIP prompt-ensemble variants
   - Source: WaffleCLIP official repository.
   - Status here: evaluated as a lightweight Waffle+PromptEns variant from cached image features.
   - Exactness: local screening variant; the existing main WaffleCLIP result remains the stronger exact-style comparison.

3. AutoCLIP-style adaptive prompt weighting
   - Source family: automatic prompt/template weighting for zero-shot CLIP.
   - Status here: evaluated as a local per-image adaptive prompt-weighting variant.
   - Exactness: screening variant, not claimed as official reproduction.

4. TPT / test-time prompt tuning
   - Source: official TPT repository.
   - Status here: code is public, but not added to final tables because it performs per-instance test-time optimization and belongs closer to TTA than static online zero-shot prompt baselines.

5. SuS-X
   - Source: official SuS-X repository.
   - Status here: code is public, but not immediately included because it requires constructing/retrieving/generating support sets and stored support features, making the setting less directly comparable to CBC's label-free no-support inference.

## Offline / transductive candidates considered

1. Sinkhorn-OT / balanced assignment
   - Source family: optimal-transport transductive calibration used in several CLIP transductive methods and already appears as a component in InMaP/Frolic-style pipelines.
   - Status here: evaluated directly on all CLIP logits.

2. PriorNorm / global predicted-prior normalization
   - Source family: test-set prior/distribution alignment and black-box logit adjustment.
   - Status here: evaluated directly on all CLIP logits.

3. LabelProp-kNN
   - Source family: ZLaP/LAME-style graph/label propagation over the unlabeled test feature set.
   - Status here: evaluated as a local kNN propagation screening baseline per corruption.
   - Exactness: screening implementation, not claimed as an official ZLaP/LAME reproduction.

4. LAME
   - Source: official LAME repository.
   - Status here: exact repo not integrated because the local screening LabelProp-kNN already provides a close graph-propagation sanity baseline; exact integration would require a separate adapter around CLIP features/logits.

5. ZLaP / Transductive CLIP-style methods
   - Status here: identified as appropriate offline candidates, but exact repositories were not integrated in this pass. If needed, these should be run as exact external baselines in a follow-up offline-only table.

## Recommendation

Use the original main online baselines for the paper's primary table. Use the extended tables for appendix or ablation-style evidence, clearly marking screening/local variants. For offline, keep InMaP/Frolic as exact strong baselines and add Sinkhorn-OT/PriorNorm as simple training-free controls; mark LabelProp-kNN as a local graph-smoothing sanity check unless replaced by exact ZLaP/LAME.
