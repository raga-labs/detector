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
