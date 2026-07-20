# Fermi-LAT 4FGL-DR4 Four-Class Source Classification

Code, data products, and manuscript for a machine-learning classification of
BL Lac / FSRQ / young pulsar / millisecond pulsar sources in the *Fermi*-LAT
4FGL-DR4 catalogue, including a covariate-shift analysis between labeled and
unassociated sources, and a probabilistic catalogue of high-confidence
candidates among the unassociated and BCU populations.

## Repository layout

```
code/           Analysis pipeline, run in numeric order (see code/README.md)
data/           Output prediction tables (probabilistic classification catalogues)
requirements.txt   Pinned Python package versions for exact reproducibility
```

## Reproducing the results

```bash
pip install -r requirements.txt
cd code
python3 01_extract_features.py     # requires gll_psc_v35.fit (4FGL-DR4, not included — see below)
python3 02_covariate_shift.py
python3 03_train_classify.py
python3 04_make_figures.py
python3 05_robustness.py
```

See `code/README.md` for a full description of each script, expected
runtime, and an important note on Random Forest determinism
(`n_jobs=1` vs `n_jobs=-1`) that affects exact reproduction of the
published per-source counts.

## Data

The input catalogue file `gll_psc_v35.fit` (4FGL-DR4) is **not** included in
this repository due to size; it is publicly available from the Fermi
Science Support Center:
<https://fermi.gsfc.nasa.gov/ssc/data/access/lat/14yr_catalog/>

The `data/` directory contains the two output catalogues produced by this
pipeline:
- `unassociated_predictions.csv` — posterior class probabilities for all
  2,423 unassociated 4FGL-DR4 sources
- `bcu_predictions.csv` — posterior class probabilities for all 1,623
  blazar-candidate-of-uncertain-type (BCU) sources



## Citation

WIll be updated

## License

Code: MIT License (see `LICENSE`).
Data products and manuscript text: all rights reserved pending journal
submission; contact the authors for reuse before publication.

## Authors

Debarshi Mukherjee, Pratik Majumdar
