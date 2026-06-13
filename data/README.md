# Data Directory

This directory is the local data landing area for `dre_production`.

Raw datasets are not committed. Place or bootstrap local datasets under:

```text
data/raw/
```

Expected public demo paths:

```text
data/raw/mimic_demo/
data/raw/openneuro/ds000030/
data/raw/chbmit/
```

Use the manifests in `data/manifests/` to see the expected lightweight structure for each dataset. Use `docs/DATA_SETUP.md` for setup and reproducibility instructions.

The `data/raw/.gitkeep` file is tracked only to preserve the raw data folder. Its contents should remain untracked.
