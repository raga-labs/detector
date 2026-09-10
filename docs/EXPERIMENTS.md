# Experiments

## E001 — One-minute vocal alapana

- Date: 2026-09-10
- Input: `data/audio/audio-1.m4a` (local/private; not committed)
- Ground truth: Mayamalavagowla
- Ground-truth tonic: A-sharp
- Duration analysed: 66.73 seconds
- TDMS windows: 5
- Detected tonic: 233.65 Hz (A-sharp 3 / B-flat 3)
- Top 1: Mayamalavagowla, 0.9167
- Top 2: Gaula, 0.0618
- Top 3: Nata, 0.0119
- Outcome: correct raga and tonic
- Dataset role: held-out real-world test; never use for training
- Independent-model result: Mayamalavagowla 0.9400, Shankarabharanam 0.0186,
  Todi 0.0134; correct

## E002 — Blind one-minute vocal alapana

- Date: 2026-09-10
- Input: `data/audio/audio-2.m4a` (local/private; not committed)
- Ground truth revealed after prediction: Mukhari
- Ground-truth tonic: A-sharp
- Duration analysed: 74.71 seconds
- TDMS windows: 5
- Detected tonic: 233.65 Hz (A-sharp 3 / B-flat 3)
- Top 1: Mukhari, 0.5999
- Top 2: Shanmukhapriya, 0.2490
- Top 3: Sindhubhairavi, 0.0835
- Outcome: correct raga and tonic in a blind test
- Dataset role: held-out real-world test; never use for training
- Independent-model result: Mukhari 0.8019, Bhairavi 0.0690, Todi 0.0255;
  correct

## E003 — Independently trained CompMusic baseline

- Date: 2026-09-10
- Data: CompMusic Carnatic pitch and tonic features
- Classes: 10 (all initial targets except Hindolam and Dharmavati)
- Split: artist-disjoint, 91 train recordings and 29 test recordings
- Training examples: 3,262 TDMS windows
- Model: XGBoost, 400 boosted rounds
- Recording-level top-1: 28/29 (0.966)
- Recording-level top-3: 28/29 (0.966)
- Error: Bhairavi by T. M. Krishna predicted as Todi
- Model path: `artifacts/models/compmusic-target-v1` (local; not committed)
- Interpretation: training pipeline validated; broader evaluation still needed

## E004 — Five-fold artist-disjoint cross-validation

- Date: 2026-09-10
- Data: all 120 CompMusic recordings for the 10 supported target ragas
- Protocol: five folds; every recording tested once; no singer leakage per fold
- Feature examples: 4,308 cached TDMS windows
- Aggregate top-1: 115/120 (0.958)
- Aggregate top-3: 119/120 (0.992)
- Fold top-1: 1.000, 0.957, 0.875, 1.000, 0.955
- Exact composition-title overlap: 2–7 titles per fold
- Private Mayamalavagowla: correct in 5/5 models; mean probability 0.937
- Private Mukhari: correct in 5/5 models; mean probability 0.842
- Metrics path: `artifacts/evaluations/compmusic-artist-cv-v1/metrics.json`
- Interpretation: accuracy is stable across unseen singers; composition-disjoint
  and larger phone-recording tests remain necessary
