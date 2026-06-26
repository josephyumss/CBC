# Additional Baseline Screening

This folder stores extra online/offline baseline candidates evaluated from cached CLIP features/logits.

## Online candidates screened
- CorruptedPrompt: generic corruption-domain prompt; diagnostic, not selected as a main paper baseline.
- PromptEns-8 / PromptEns-80: standard CLIP prompt feature averaging with selected/all templates.
- LogitEns-8 / LogitEns-80: prompt logit averaging variants.
- AdaptivePromptEns-80: AutoCLIP-style per-image prompt weighting; local lightweight implementation.
- Waffle+PromptEns: Waffle-style descriptors combined with CLIP prompt ensemble.

## Offline candidates screened
- Sinkhorn-OT: balanced optimal-transport assignment on all test logits.
- PriorNorm: global predicted-prior normalization.
- LabelProp-kNN: local ZLaP/LAME-style kNN label propagation screening implementation.

## Selection rule
Candidates were added to extended tables when they are plausible, training-free, code-light baselines and do not use labels. Exact official reproduction status is noted in metadata/notes; paper tables should distinguish exact baselines from local screening variants.
