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
