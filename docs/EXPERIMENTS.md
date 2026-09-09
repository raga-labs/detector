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
