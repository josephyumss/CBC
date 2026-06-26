# CBC in Offline Setting

- Compared methods: CLIP, CBC Offline, InMaP, Frolic.
- CBC Offline uses one global bias/center from the entire evaluated test split, rather than batch-wise centering.
- Important distinction:
  - `offline_setting_overall_s5.csv` is the mixed-corruption global-center setting over all 150k severity-5 logits.
  - `offline_setting_corruptionwise_macro_s5.csv` is the proper corruption-wise macro setting: compute one global CBC bias per corruption, then average over 15 corruptions.
- Files: `offline_setting_overall_s5.csv`, `offline_setting_by_corruption_s5.csv`, `offline_setting_corruptionwise_macro_s5.csv`, `offline_vs_online_cbc_by_corruption_s5.csv`, `offline_cbc_definition.json`, `offline_cbc_corruptionwise_note.json`.
- Mixed S5 global-center: CLIP 60.191 → CBC Offline 64.978; InMaP 67.225; Frolic 66.761.
- Corruption-wise S5 macro rerun: CLIP 60.191 → CBC Offline 66.864; InMaP 69.526; Frolic 68.721.
- Extended offline candidate table: `offline_setting_overall_s5_extended.csv`.
