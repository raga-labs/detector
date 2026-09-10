# Model A baseline

## Decision

Use a tonic-normalized Time-Delayed Melody Surface (TDMS) over predominant-pitch
tracks, followed by a compact classifier. TDMS captures pitch movement and
ornamentation rather than only the set of notes.

We will first reproduce and audit the open-source TwelveSwaras XGBoost baseline,
which uses the same CompMusic features. Its code is MIT-licensed; its published
seed model is non-commercial because of its training-data licences.

Reference: <https://github.com/twelveswaras/twelveswaras>

## Initial coverage

CompMusic's 40-raga Carnatic set contains 10 of our requested 12 ragas:

- Mayamalavagowla
- Kharaharapriya
- Kalyani
- Mohanam
- Shankarabharanam
- Bhairavi
- Mukhari
- Saveri
- Kambhoji
- Todi

Hindolam and Dharmavati require additional labelled data. They stay in the
product vocabulary but will not be claimed as supported by the first model.

## Independent baseline result

Our first model was trained from scratch on the CompMusic pitch and tonic
features for the 10 supported ragas. The deterministic artist-disjoint split
used 91 recordings for training and 29 recordings for testing; no test singer
appeared in training.

- Training windows: 3,262
- Recording-level top-1: 28/29 (0.966)
- Recording-level top-3: 28/29 (0.966)
- Error: a T. M. Krishna Bhairavi recording was classified as Todi
- Private phone tests: Mayamalavagowla and Mukhari both correct

This is strong evidence that our training implementation works, not yet a
production-quality accuracy claim. The test set is small, represents concert
recordings rather than phone alapanas, and may contain compositions also heard
in training under different artists. Multi-split evaluation and a larger
real-world blind set are required next.

## Five-fold artist-disjoint evaluation

Every one of the 120 recordings was held out exactly once across five folds.
In each fold, none of its test singers appeared in that fold's training data.

- Aggregate top-1: 115/120 (0.958)
- Aggregate top-3: 119/120 (0.992)
- Fold top-1 range: 0.875–1.000
- TDMS windows cached across folds: 4,308
- Exact normalized composition-title overlaps by fold: 2, 2, 7, 5, 6

Top-1 errors were two Mukhari-to-Bhairavi confusions, one
Bhairavi-to-Mukhari confusion, one Bhairavi-to-Todi confusion, and one
Kambhoji-to-Mohanam confusion. Only the Bhairavi-to-Todi error fell outside the
top three.

Both private phone recordings were correct in all five models. Mean raw
probability was 0.937 for Mayamalavagowla and 0.842 for Mukhari. The remaining
composition-title overlap means this is strong validation, but not yet a fully
composition-independent or real-world production benchmark.
