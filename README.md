# Raga Identification — Model A

Pitch-first classifier for identifying a Carnatic raga from an approximately
one-minute alapana.

## Initial scope

The first model is a closed-set classifier for the 12 ragas in
`config/ragas.json`. More ragas can be added later by adding diverse labelled
recordings and retraining the classifier.

## Data layout

Place audio under `data/audio/` and describe every recording in
`data/manifest.csv`. Keep recordings from the same singer and session together
under one `group_id`; this prevents train/test leakage.

The initial pipeline will:

1. extract the predominant pitch;
2. normalize it relative to the tonic (Sa);
3. build Time-Delayed Melody Surface (TDMS) windows;
4. train a compact classifier;
5. aggregate window predictions for the one-minute result.

The first independent CompMusic experiment supports 10 of the 12 target ragas;
Hindolam and Dharmavati are pending suitable training data.

See `docs/BASELINE.md` for the baseline decision and dataset coverage.

## Local inference

After downloading the reference model and installing `requirements-inference.txt`:

```bash
raga-detect path/to/alapana.wav
```

The command returns the detected tonic and top three ragas as JSON.
Phone-recorded M4A files require FFmpeg (`brew install ffmpeg` on macOS).

## Train the independent baseline

After downloading and extracting the CompMusic feature archive:

```bash
.venv/bin/python scripts/train_compmusic.py
```

The split is artist-disjoint: no singer in the test recordings appears in the
training recordings. Generated models and metrics are kept under `artifacts/`
and are not committed.

Run the stricter five-fold artist-disjoint evaluation with:

```bash
.venv/bin/python scripts/cross_validate_compmusic.py
```
