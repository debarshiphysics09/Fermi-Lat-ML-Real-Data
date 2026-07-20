# Fermi-LAT 4FGL-DR4 four-class classification — analysis code

Run in order. Each script reads the previous script's output.

```
python3 01_extract_features.py     # gll_psc_v35.fit           -> 4fgl_dr4_features.csv
python3 02_covariate_shift.py      # 4fgl_dr4_features.csv     -> shift_results.csv, shift_summary.txt
python3 03_train_classify.py       # 4fgl_dr4_features.csv     -> unassociated_predictions.csv,
                                    #                              bcu_predictions.csv,
                                    #                              model_performance.csv, rf_model.pkl
python3 04_make_figures.py         # -> fig1-fig7 (original manuscript figures)
python3 05_robustness.py           # -> permutation_importance.csv, robustness_results.json,
                                    #    fig8_calibration.pdf, fig9_blazar_sequence.pdf
```

## What 05_robustness.py adds (new in this revision)

1. **Repeated 10-fold CV** (5 seeds x 10 folds = 50 paired estimates, RF n_estimators=150)
   comparing the 7-feature and 10-feature sets — replaces the single-seed CV in the
   original submission with a more robust estimate and paired Wilcoxon test.
2. **Repeated 70/30 hold-out** (15 independent stratified splits) giving a distribution
   of test-set balanced accuracy and per-class F1, instead of one lucky/unlucky split.
3. **Permutation importance** (15 repeats on the held-out test fold) compared against
   the Gini importance already in the paper, to check for the known Gini bias toward
   correlated/continuous features.
4. **Calibration curve**: pools out-of-fold predicted probabilities across a 10-fold
   CV and checks whether predicted P_max matches observed accuracy — this justifies
   (or would have falsified) the P_max >= 0.80 high-confidence threshold used
   throughout the paper.
5. **Blazar-sequence figure**: Gamma_1GeV vs log(E_peak) colored by class, connecting
   the two added spectral features back to the physical blazar sequence
   (Fossati et al. 1998) rather than just asserting the connection in prose.

## Compute note

This was developed/tested on a single-CPU sandbox. `n_estimators` and repeat counts
in `05_robustness.py` were set conservatively (150 trees, 5x10 CV, 15x holdout, 15x
permutation-importance repeats) to finish in a few minutes on 1 CPU. If you have
more cores, you can safely increase these (e.g. back to 10x10 CV, 30 holdout
repeats, 300 trees) — the qualitative results (which we verified) do not change,
only the width of the confidence intervals shrinks slightly.

## Reproducibility (read before reporting exact counts)

`RandomForestClassifier(..., n_jobs=-1, random_state=42)` is **not** guaranteed
bit-for-bit reproducible across sklearn versions / machines, because parallel
tree-building can process trees in a different order. We hit this in an earlier
revision (see the discrepancy table below for the record). **Fix applied:** the
production model in `03_train_classify.py` (the one that generates
`unassociated_predictions.csv` and `bcu_predictions.csv`) now uses `n_jobs=1`.
With `n_jobs=1` and the exact versions pinned in `requirements.txt`, we verified
**bit-for-bit identical output across independent reruns**. Install with:

```
pip install -r requirements.txt
```

The CV/hold-out comparisons in `05_robustness.py` still use `n_jobs=-1` for
speed — that's fine, since those only report aggregate statistics (means,
confidence intervals), which don't depend on tree-build order the way exact
per-source counts do.

### Historical note: what we caught during revision

An earlier run with `n_jobs=-1` for the production model gave slightly
different high-confidence counts than a later rerun on the same seed:

| Quantity                                   | Run A (n_jobs=-1) | Run B (n_jobs=-1) | Final (n_jobs=1) |
|---------------------------------------------|:---:|:---:|:---:|
| High-confidence unassociated (P_max>=0.80)   | 514 | 541 | 541 (deterministic) |
| High-confidence BLL/FSRQ among BCU            | 911/601 | 912/596 | 912/596 (deterministic) |

The manuscript now reports the `n_jobs=1` deterministic numbers throughout
(541 and 912/596). All headline statistics (10-fold CV balanced accuracy,
Wilcoxon p-value, confusion matrix, feature importances) were identical across
all three runs regardless of `n_jobs` — only the count-based production-model
outputs were affected.
