# Robustness of CBC

- Compared methods: CLIP, CLIP + CBC, CALIP, WaffleCLIP, CuPL.
- `corruption_mixed_s5.csv`: all CIFAR-10-C severity 5 corruptions mixed.
- `severity_mixed_s1_s5.csv`: severity 1–5 mixed; CLIP/CBC from actual mixed run, other online baselines from equal-weight severity averages.
- `batch_size_sweep_online.csv`: batch sensitivity; only CBC is batch-dependent.
- `class_imbalance_*_online.csv`: corruption-wise class imbalance sweep 0–100%.
