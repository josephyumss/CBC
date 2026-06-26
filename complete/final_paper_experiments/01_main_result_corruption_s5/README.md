# Main Result: CLIP Zero-shot Performance under Corruption

- Dataset/setting: CIFAR-10-C severity 5, 15 corruptions, ViT-B/16.
- Compared methods: CLIP, CLIP + CBC, CALIP, WaffleCLIP, CuPL.
- Offline/transductive methods InMaP and Frolic are intentionally excluded here.
- Overall: CLIP 60.191 → CBC 64.729.
- Files: `main_corruption_s5_long.csv`, `main_corruption_s5_wide.csv`, `main_corruption_s5_overall.csv`.
- Extended candidate table: `main_corruption_s5_overall_extended_online.csv`.
