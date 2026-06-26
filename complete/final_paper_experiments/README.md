# Final Paper Experiments

논문 최종 실험 흐름에 맞춰 기존 완료 결과와 full-suite 결과를 재정리한 폴더입니다.

## Sections
- `01_main_result_corruption_s5/`: online zero-shot methods under CIFAR-10-C severity 5 corruptions.
- `02_robustness_online/`: corruption mixed, severity mixed, class imbalance, batch-size sweep.
- `03_offline_setting/`: CLIP, CBC Offline, InMaP, Frolic.
- `04_clean_image/`: clean CIFAR-10 with online methods plus CBC Offline.
- `05_qualitative_analysis/`: sample grid and PCA visualizations.
- `06_cbc_with_tta_methods/`: CLIP + 9 TTA raw + 9 TTA with CBC.

## Baseline Split
- Online/non-transductive zero-shot baselines: CLIP, CBC, CALIP, WaffleCLIP, CuPL.
- Offline/transductive baselines: CBC Offline, InMaP, Frolic.
- InMaP/Frolic are intentionally separated from online main/robustness tables because they use the full test feature set.
