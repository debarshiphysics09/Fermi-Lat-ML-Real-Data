#!/usr/bin/env python3
"""
03_train_classify.py
====================
Step 3: Train four-class classifiers (BLL/FSRQ/PSR/MSP) on the real
4FGL-DR4 labeled sample and classify unassociated + BCU sources.

Key operations
--------------
1. Load features from 01_extract_features.py output
2. Compare 7-feature baseline (Malyshev & Bhat 2023) vs 10-feature set
   using 10-fold stratified cross-validation with paired Wilcoxon test
3. Train final Random Forest on full labeled sample
4. Classify all 2423 unassociated sources and 1622 BCU sources
5. Save posterior probability tables as CSV

Input:   4fgl_dr4_features.csv   (from 01_extract_features.py)
         shift_auc.txt            (from 02_covariate_shift.py)
Outputs: unassociated_predictions.csv
         bcu_predictions.csv
         model_performance.csv
         rf_model.pkl             (trained model + scaler, for inspection)
"""

import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                      cross_val_score, cross_validate)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (balanced_accuracy_score, accuracy_score,
                              roc_auc_score, confusion_matrix,
                              precision_recall_fscore_support)
from scipy.stats import wilcoxon

SEED = 42
CN   = ['BLL', 'FSRQ', 'PSR', 'MSP']   # class names (ClassID 0,1,2,3)

# ── Feature sets ──────────────────────────────────────────────────────
# 7-feature baseline: exactly Malyshev & Bhat (2023)
FEAT_M7 = [
    'Log_Energy_Flux', 'Log_Unc_Flux', 'Log_Signif',
    'LP_index1GeV', 'LP_beta', 'LP_SigCurv', 'Log_Var',
]

# 10-feature set: this work (adds GLAT_abs, Log_LP_EPeak_imp, Frac_Variability)
FEAT_10 = [
    'Log_Energy_Flux', 'Log_Unc_Flux', 'Log_Signif',
    'LP_index1GeV', 'LP_beta', 'LP_SigCurv', 'Log_Var',
    'GLAT_abs', 'Log_LP_EPeak_imp', 'Frac_Variability',
]


# ── Helpers ───────────────────────────────────────────────────────────
def load_data(features_csv):
    df = pd.read_csv(features_csv)
    df['IsLabeled'] = df['IsLabeled'].astype(bool)
    df['IsUnassoc'] = df['IsUnassoc'].astype(bool)
    df['IsBCU']     = df['IsBCU'].astype(bool)
    medians = df[df.IsLabeled][FEAT_10].median()
    for feat in FEAT_10:
        df[feat] = df[feat].fillna(medians[feat])
    return df


def print_section(title):
    print()
    print("=" * 65)
    print(title)
    print("=" * 65)


def eval_metrics(y_true, y_pred, y_prob, label=''):
    acc  = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    cm   = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0,1,2,3], average=None)
    if label:
        print(f"\n  {label}")
        print(f"    Acc={100*acc:.1f}%  BalAcc={100*bacc:.1f}%  AUC={auc:.3f}  "
              f"MacroF1={f1.mean():.3f}")
        print(f"    Per-class F1: "
              + "  ".join(f"{c}:{v:.3f}" for c,v in zip(CN,f1)))
    return dict(Acc=acc, BalAcc=bacc, AUC=auc, CM=cm,
                Prec=pr, Rec=rc, F1=f1, MacroF1=f1.mean(),
                y_pred=y_pred, y_prob=y_prob)


# ── Main pipeline ─────────────────────────────────────────────────────
def run(features_csv='4fgl_dr4_features.csv'):

    df = load_data(features_csv)

    df_lab  = df[df.IsLabeled].copy()
    df_unas = df[df.IsUnassoc].copy()
    df_bcu  = df[df.IsBCU].copy()

    X7   = df_lab[FEAT_M7].values
    X10  = df_lab[FEAT_10].values
    y    = df_lab['ClassID'].values.astype(int)

    print_section("DATASET SUMMARY")
    print(f"  Labeled (train/test): {len(df_lab)}")
    for c, cid in zip(CN, range(4)):
        print(f"    {c}: {(y==cid).sum()} ({100*(y==cid).mean():.1f}%)")
    print(f"  Unassociated (predict): {len(df_unas)}")
    print(f"  BCU (predict):          {len(df_bcu)}")

    # ── STEP A: 10-fold CV feature-set comparison ─────────────────────
    print_section("STEP A: FEATURE-SET COMPARISON (10-fold CV, RF n=300)")

    cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    rf_cv = RandomForestClassifier(
        n_estimators=300, max_features='sqrt', min_samples_leaf=2,
        class_weight='balanced', n_jobs=-1, random_state=SEED)

    sc7  = StandardScaler().fit(X7)
    sc10 = StandardScaler().fit(X10)

    s7   = cross_val_score(rf_cv, sc7.transform(X7),   y, cv=cv10,
                           scoring='balanced_accuracy')
    s10  = cross_val_score(rf_cv, sc10.transform(X10), y, cv=cv10,
                           scoring='balanced_accuracy')
    _, pw = wilcoxon(s10, s7)

    print(f"\n  7-feature (Malyshev & Bhat 2023 baseline):")
    print(f"    CV BalAcc = {100*s7.mean():.1f} +/- {100*s7.std():.1f}%")
    print(f"    Folds: {[f'{100*v:.1f}' for v in s7]}")
    print(f"\n  10-feature (this work):")
    print(f"    CV BalAcc = {100*s10.mean():.1f} +/- {100*s10.std():.1f}%")
    print(f"    Folds: {[f'{100*v:.1f}' for v in s10]}")
    print(f"\n  Delta = +{100*(s10.mean()-s7.mean()):.1f}%")
    print(f"  Wilcoxon signed-rank p = {pw:.4f} "
          f"({'significant' if pw < 0.05 else 'not significant'} at alpha=0.05)")

    # ── STEP B: 70/30 held-out test set evaluation ────────────────────
    print_section("STEP B: HELD-OUT TEST SET (70/30, RF n=500)")

    X10_tr, X10_te, y_tr, y_te = train_test_split(
        X10, y, test_size=0.30, stratify=y, random_state=SEED)
    X7_tr,  X7_te,  _,    _   = train_test_split(
        X7,  y, test_size=0.30, stratify=y, random_state=SEED)
    print(f"  Train: {len(y_tr)}   Test: {len(y_te)}")

    def fit_eval(X_tr, X_te, clf, label):
        sc = StandardScaler().fit(X_tr)
        clf.fit(sc.transform(X_tr), y_tr)
        yp  = clf.predict(sc.transform(X_te))
        ypr = clf.predict_proba(sc.transform(X_te))
        return eval_metrics(y_te, yp, ypr, label), sc, clf

    rf500 = RandomForestClassifier(
        n_estimators=500, max_features='sqrt', min_samples_leaf=2,
        class_weight='balanced', n_jobs=-1, random_state=SEED)
    rf7_500 = RandomForestClassifier(
        n_estimators=500, max_features='sqrt', min_samples_leaf=2,
        class_weight='balanced', n_jobs=-1, random_state=SEED)
    gb = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.08, max_depth=4,
        subsample=0.8, random_state=SEED)
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam',
        alpha=1e-4, batch_size=64, learning_rate='adaptive', max_iter=1000,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
        random_state=SEED)

    res10_rf, sc10_final, rf10_final = fit_eval(
        X10_tr, X10_te, rf500,   '10-feat RF  (n=500)')
    res7_rf,  sc7_tmp,  rf7_tmp   = fit_eval(
        X7_tr,  X7_te,  rf7_500, ' 7-feat RF  (n=500)')
    res10_gb, sc10_gb, gb_final   = fit_eval(
        X10_tr, X10_te, gb,      '10-feat GB')
    res10_mlp,sc10_mlp,mlp_final  = fit_eval(
        X10_tr, X10_te, mlp,     '10-feat MLP')

    # Per-class F1 delta
    delta_f1 = res10_rf['F1'] - res7_rf['F1']
    print(f"\n  Per-class F1 improvement (10-feat RF vs 7-feat RF):")
    for c, d in zip(CN, delta_f1):
        print(f"    {c}: {d:+.3f}")

    # Confusion matrix
    print(f"\n  Confusion matrix (10-feat RF, rows=true, cols=pred):")
    print(f"  {'':>6}", "  ".join(f"{c:>6}" for c in CN))
    for i, c in enumerate(CN):
        row = res10_rf['CM'][i]
        print(f"  {c:>6}", "  ".join(f"{v:>6}" for v in row))

    # Feature importances (RF)
    fi = rf10_final.feature_importances_
    print(f"\n  Feature importances (10-feat RF, Gini):")
    for feat, imp in sorted(zip(FEAT_10, fi), key=lambda x: -x[1]):
        print(f"    {feat:<24} {100*imp:.1f}%")

    # Save performance summary
    perf_rows = []
    for label, res, n_feat in [
        ('RF_7feat',  res7_rf,   7),
        ('RF_10feat', res10_rf, 10),
        ('GB_10feat', res10_gb, 10),
        ('MLP_10feat',res10_mlp,10),
    ]:
        perf_rows.append({
            'Model': label, 'N_features': n_feat,
            'Accuracy': round(100*res['Acc'],1),
            'BalAcc':   round(100*res['BalAcc'],1),
            'AUC':      round(res['AUC'],3),
            'MacroF1':  round(res['MacroF1'],3),
            **{f'F1_{c}': round(v,3) for c,v in zip(CN, res['F1'])},
        })
    pd.DataFrame(perf_rows).to_csv('model_performance.csv', index=False)
    print(f"\n  Performance table saved: model_performance.csv")

    # ── STEP C: Retrain on FULL labeled sample + classify ─────────────
    print_section("STEP C: PRODUCTION MODEL (full labeled sample)")

    sc_prod = StandardScaler().fit(X10)
    rf_prod = RandomForestClassifier(
        n_estimators=500, max_features='sqrt', min_samples_leaf=2,
        class_weight='balanced', n_jobs=1, random_state=SEED)
    rf_prod.fit(sc_prod.transform(X10), y)
    print(f"  Trained RF on {len(y)} sources (all 4 classes)")

    # Read OOD threshold
    ts_p5 = df_lab['Log_Signif'].quantile(0.05)

    def classify_set(df_tgt, label):
        X_tgt = sc_prod.transform(df_tgt[FEAT_10].values)
        probs  = rf_prod.predict_proba(X_tgt)
        pred   = probs.argmax(axis=1)
        conf   = probs.max(axis=1)
        out = df_tgt[['Source_Name','GLON','GLAT','CLASS1']].copy()
        for i, c in enumerate(CN):
            out[f'P_{c}'] = probs[:, i]
        out['PredClass']  = [CN[p] for p in pred]
        out['Confidence'] = conf
        if 'Log_Signif' in df_tgt.columns:
            out['IsOOD'] = (df_tgt['Log_Signif'].values < ts_p5)
        print(f"\n  {label} (N={len(out)}):")
        for c in CN:
            n = (out.PredClass == c).sum()
            print(f"    {c}: {n:>5} ({100*n/len(out):.1f}%)")
        if 'IsOOD' in out.columns:
            hc = out[(out.Confidence >= 0.80) & (~out.IsOOD)]
            print(f"  High-confidence non-OOD (P_max >= 0.80): {len(hc)}")
            for c in CN:
                print(f"    {c}: {(hc.PredClass==c).sum()}")
        return out

    out_unas = classify_set(df_unas, 'Unassociated sources')
    out_unas.to_csv('unassociated_predictions.csv', index=False)
    print(f"\n  Saved: unassociated_predictions.csv")

    out_bcu = classify_set(df_bcu, 'BCU sources')
    out_bcu.to_csv('bcu_predictions.csv', index=False)
    print(f"  Saved: bcu_predictions.csv")

    # Save model for reproducibility
    with open('rf_model.pkl', 'wb') as f:
        pickle.dump({'rf': rf_prod, 'sc': sc_prod,
                     'features': FEAT_10, 'classes': CN,
                     'ood_threshold': ts_p5,
                     'cv_bacc_7feat': (s7.mean(), s7.std()),
                     'cv_bacc_10feat': (s10.mean(), s10.std()),
                     'wilcoxon_p': pw}, f)
    print(f"  Saved: rf_model.pkl (trained model + scaler)")

    print_section("DONE")
    print(f"  Key result: 10-feat CV BalAcc = {100*s10.mean():.1f}% "
          f"vs 7-feat {100*s7.mean():.1f}%  (p={pw:.3f})")
    print(f"  Test-set:  RF BalAcc={100*res10_rf['BalAcc']:.1f}%  "
          f"AUC={res10_rf['AUC']:.3f}")


if __name__ == '__main__':
    run('4fgl_dr4_features.csv')
