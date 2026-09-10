# Dataset

## CompMusic raga-recognition features

Source: <https://zenodo.org/records/7278506>

The archive contains tonic, pitch, nyas, tani, and metadata features for the
Carnatic and Hindustani raga-recognition datasets. Raw dataset files are not
committed to Git.

Download with:

```bash
bash scripts/download_compmusic_features.sh
```

After download, the next pipeline step will select the Carnatic subset, map
available labels to `config/ragas.json`, and report which of our 12 initial
ragas are represented.

## Saraga metadata audit

Saraga's Git repository contains metadata and download helpers, but not the
large pitch or audio files. Reproduce the local metadata audit with:

```bash
bash scripts/download_saraga_metadata.sh
python3 scripts/audit_saraga_metadata.py data/raw/saraga-metadata
```

The 2026-09-10 audit found the following labelled recording counts:

| Raga | Recordings |
| --- | ---: |
| Mayamalavagowla | 0 |
| Kharaharapriya | 2 |
| Kalyani | 4 |
| Hindolam | 1 |
| Mohanam | 4 |
| Shankarabharanam | 3 |
| Bhairavi | 5 |
| Mukhari | 0 |
| Saveri | 3 |
| Kambhoji | 3 |
| Dharmavati | 0 |
| Todi | 7 |

Saraga therefore provides useful bootstrap material, but only 32 recordings
for our target set and cannot support a robust 12-class model by itself.
Pitch and tonic annotations can be selected through the Dunya API using the
`pitch` and `ctonic` slugs, but that route requires a Dunya API token. The full
archive is also available through Zenodo when that service is reachable.

## CompMusic target coverage

The verified 3.6 GB feature archive was downloaded from Zenodo and only its
Carnatic directory was extracted. It contains 12 pitch/tonic recordings for
each of 10 target ragas, with 8–12 distinct artists per raga:

Mayamalavagowla, Kharaharapriya, Kalyani, Mohanam, Shankarabharanam, Bhairavi,
Mukhari, Saveri, Kambhoji, and Todi.

Hindolam and Dharmavati are absent. The first independently trained model is
therefore a truthful 10-class experiment. They will be added only after suitable
training data is obtained.
