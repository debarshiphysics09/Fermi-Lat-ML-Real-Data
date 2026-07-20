#!/usr/bin/env python3
"""
04_make_figures.py
==================
Step 4: Generate all publication-quality figures for the manuscript.

Input:   4fgl_dr4_features.csv          (from 01_extract_features.py)
         rf_model.pkl                    (from 03_train_classify.py)
         unassociated_predictions.csv    (from 03_train_classify.py)
         bcu_predictions.csv             (from 03_train_classify.py)
Output:  figures/fig1_parameter_space.pdf
         figures/fig2_feature_distributions.pdf
         figures/fig3_covariate_shift.pdf
         figures/fig4_feature_comparison.pdf
         figures/fig5_cm_roc.pdf
         figures/fig6_feature_importance.pdf
         figures/fig7_sky_probabilities.pdf
"""

import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

from scipy.stats import ks_2samp, wasserstein_distance, wilcoxon
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                      cross_val_score, cross_validate)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_curve, auc as roc_auc_,
                              balanced_accuracy_score, confusion_matrix,
                              precision_recall_fscore_support,
                              roc_auc_score)

FIGDIR = 'figures'
os.makedirs(FIGDIR, exist_ok=True)
SEED = 42
CN   = ['BLL', 'FSRQ', 'PSR', 'MSP']
CC   = ['#2166AC', '#D6604D', '#1A9850', '#762A83']

FEAT_M7 = ['Log_Energy_Flux','Log_Unc_Flux','Log_Signif',
           'LP_index1GeV','LP_beta','LP_SigCurv','Log_Var']
FEAT_10 = ['Log_Energy_Flux','Log_Unc_Flux','Log_Signif',
           'LP_index1GeV','LP_beta','LP_SigCurv','Log_Var',
           'GLAT_abs','Log_LP_EPeak_imp','Frac_Variability']
FEAT_LABELS = [
    r'$\log F_E$', r'$\log\sigma_{F}$', r'$\log\sqrt{TS}$',
    r'$\Gamma_{\rm 1\,GeV}$', r'$\beta_{\rm LP}$',
    r'$\sigma_{\rm curv}$', r'$\log V$',
    r'$|b|$ (°)', r'$\log E^{\rm LP}_{\rm peak}$', r'$F_{\rm var}$',
]

# ── Style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Serif', 'font.size': 11,
    'axes.labelsize': 11, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'figure.dpi': 150,
    'axes.linewidth': 1.0,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
})


def savefig(name, dpi=300):
    path = os.path.join(FIGDIR, name)
    plt.savefig(path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close('all')
    print(f'  Saved: {name}')


def load_all(features_csv='4fgl_dr4_features.csv',
             model_pkl='rf_model.pkl'):
    df = pd.read_csv(features_csv)
    df['IsLabeled'] = df['IsLabeled'].astype(bool)
    df['IsUnassoc'] = df['IsUnassoc'].astype(bool)
    df['IsBCU']     = df['IsBCU'].astype(bool)
    medians = df[df.IsLabeled][FEAT_10].median()
    for feat in FEAT_10:
        df[feat] = df[feat].fillna(medians[feat])

    with open(model_pkl, 'rb') as f:
        M = pickle.load(f)

    df_u = pd.read_csv('unassociated_predictions.csv')
    df_b = pd.read_csv('bcu_predictions.csv')
    return df, M, df_u, df_b


def make_all_figures(features_csv='4fgl_dr4_features.csv',
                     model_pkl='rf_model.pkl'):

    df, M, df_u, df_b = load_all(features_csv, model_pkl)
    df_lab = df[df.IsLabeled].copy()
    df_unas= df[df.IsUnassoc].copy()
    y_all  = df_lab['ClassID'].values.astype(int)
    sc     = M['sc'];  rf = M['rf']
    ts_p5  = M['ood_threshold']

    # Train/test split (same seed as 03_train_classify.py)
    X10 = sc.transform(df_lab[FEAT_10].values)
    X7  = StandardScaler().fit_transform(df_lab[FEAT_M7].values)
    X10_tr, X10_te, y_tr, y_te = train_test_split(
        X10, y_all, test_size=0.30, stratify=y_all, random_state=SEED)
    X7_tr,  X7_te,  _,    _   = train_test_split(
        X7,  y_all, test_size=0.30, stratify=y_all, random_state=SEED)

    rf10 = RandomForestClassifier(n_estimators=500, max_features='sqrt',
                                   min_samples_leaf=2, class_weight='balanced',
                                   n_jobs=-1, random_state=SEED)
    rf7  = RandomForestClassifier(n_estimators=500, max_features='sqrt',
                                   min_samples_leaf=2, class_weight='balanced',
                                   n_jobs=-1, random_state=SEED)
    rf10.fit(X10_tr, y_tr); rf7.fit(X7_tr, y_tr)
    yp10 = rf10.predict(X10_te); ypr10 = rf10.predict_proba(X10_te)
    yp7  = rf7.predict(X7_te);   ypr7  = rf7.predict_proba(X7_te)

    print("Generating figures...")

    # ── Figure 1: parameter space (4 panels) ─────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.0))
    panels = [
        ('LP_index1GeV', 'LP_SigCurv',
         r'$\Gamma_{\rm 1\,GeV}$ (Spectral index at 1 GeV)',
         r'$\sigma_{\rm curv}$ (Curvature significance)'),
        ('LP_index1GeV', 'LP_beta',
         r'$\Gamma_{\rm 1\,GeV}$',
         r'$\beta_{\rm LP}$ (Curvature parameter)'),
        ('GLAT_abs', 'LP_SigCurv',
         r'$|b|$ (Galactic latitude, °)',
         r'$\sigma_{\rm curv}$'),
        ('Log_Var', 'Frac_Variability',
         r'$\log V$ (Variability index)',
         r'$F_{\rm var}$ (Fractional variability)'),
    ]
    for ax, (fx, fy, xl, yl), lbl in zip(
            axes.ravel(), panels, ['(a)', '(b)', '(c)', '(d)']):
        for cid, (cls, col) in enumerate(zip(CN, CC)):
            m = (y_all == cid)
            ax.scatter(df_lab[fx][m], df_lab[fy][m], c=col, alpha=0.25,
                       s=3, label=cls, rasterized=True, linewidths=0)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(lbl, loc='left', fontweight='bold')
        ax.legend(markerscale=4, handletextpad=0.3, framealpha=0.8,
                  loc='upper right')
    plt.tight_layout(pad=0.8)
    savefig('fig1_parameter_space.pdf')

    # ── Figure 2: feature distributions ──────────────────────────────
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.5))
    for ax, feat, flbl in zip(axes.ravel(), FEAT_10, FEAT_LABELS):
        for cid, (cls, col) in enumerate(zip(CN, CC)):
            v = df_lab[feat][y_all == cid].values
            ax.hist(v, bins=30, density=True, alpha=0.55, color=col,
                    edgecolor='none', label=cls)
            ax.axvline(np.median(v), color=col, lw=1.2, ls='--', alpha=0.9)
        ax.set_xlabel(flbl, fontsize=9)
        ax.set_ylabel('Density', fontsize=8)
        ax.tick_params(labelsize=8)
    axes[0, 0].legend(fontsize=7.5, markerscale=1.5, framealpha=0.7)
    plt.suptitle('Feature distributions (dashed lines = class medians)',
                 fontsize=11, y=1.01)
    plt.tight_layout(pad=0.5)
    savefig('fig2_feature_distributions.pdf')

    # ── Figure 3: covariate shift ─────────────────────────────────────
    fig = plt.figure(figsize=(13, 8))
    gs  = gridspec.GridSpec(2, 5, figure=fig, hspace=0.55, wspace=0.45)
    axes_sh = [fig.add_subplot(gs[r, c])
               for r in range(2) for c in range(5)]
    for ax, feat, flbl in zip(axes_sh, FEAT_10, FEAT_LABELS):
        lv = df_lab[feat].values
        uv = df_unas[feat].values
        vmin = min(np.percentile(lv, 1), np.percentile(uv, 1))
        vmax = max(np.percentile(lv, 99), np.percentile(uv, 99))
        bins = np.linspace(vmin, vmax, 35)
        ax.hist(lv, bins=bins, density=True, alpha=0.65, color='#2166AC',
                edgecolor='none', label='Labeled')
        ax.hist(uv, bins=bins, density=True, alpha=0.65, color='#D6604D',
                edgecolor='none', label='Unassoc.')
        wd = wasserstein_distance(lv, uv)
        ks, _ = ks_2samp(lv, uv)
        ax.set_xlabel(flbl, fontsize=8.5)
        ax.set_ylabel('Density', fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.set_title(rf'$W_1={wd:.2f}$, $D_{{\rm KS}}={ks:.3f}$', fontsize=8)
    axes_sh[0].legend(fontsize=7.5, loc='upper right')
    fig.suptitle(
        'Covariate shift: labeled vs. unassociated sources',
        fontsize=10.5, y=1.005)
    savefig('fig3_covariate_shift.pdf')

    # ── Figure 4: feature-set comparison ─────────────────────────────
    cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    rf_cv = RandomForestClassifier(n_estimators=300, max_features='sqrt',
                                    min_samples_leaf=2, class_weight='balanced',
                                    n_jobs=-1, random_state=SEED)
    gb_cv = GradientBoostingClassifier(n_estimators=200, learning_rate=0.08,
                                        max_depth=4, subsample=0.8,
                                        random_state=SEED)
    mlp_cv = MLPClassifier(hidden_layer_sizes=(128, 64, 32), solver='adam',
                            alpha=1e-4, batch_size=64, max_iter=1000,
                            early_stopping=True, validation_fraction=0.15,
                            n_iter_no_change=20, random_state=SEED)

    results = {}
    for fname, feats in [('7-feature\n(Malyshev\n& Bhat 2023)', FEAT_M7),
                          ('10-feature\n(this work)', FEAT_10)]:
        Xf = StandardScaler().fit_transform(df_lab[feats].values)
        for cname, clf in [('RF', rf_cv), ('GB', gb_cv), ('MLP', mlp_cv)]:
            s = cross_val_score(clf, Xf, y_all, cv=cv10,
                                scoring='balanced_accuracy')
            results[(fname, cname)] = s

    s7cv  = results[('7-feature\n(Malyshev\n& Bhat 2023)', 'RF')]
    s10cv = results[('10-feature\n(this work)', 'RF')]
    _, pw = wilcoxon(s10cv, s7cv)

    pr7,  rc7,  f1_7,  _ = precision_recall_fscore_support(
        y_te, yp7,  labels=[0,1,2,3], average=None)
    pr10, rc10, f1_10, _ = precision_recall_fscore_support(
        y_te, yp10, labels=[0,1,2,3], average=None)
    delta_f1 = f1_10 - f1_7

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    feat_names = ['7-feature\n(Malyshev\n& Bhat 2023)', '10-feature\n(this work)']
    clf_names  = ['RF', 'GB', 'MLP']
    clrs_clf   = ['#4393C3', '#F4A582', '#92C5DE']
    x = np.arange(len(feat_names)); width = 0.22

    ax = axes[0]
    for ci, (cn, clr) in enumerate(zip(clf_names, clrs_clf)):
        means = [100*results[(fn, cn)].mean() for fn in feat_names]
        stds  = [100*results[(fn, cn)].std()  for fn in feat_names]
        ax.bar(x + (ci-1)*width, means, width, yerr=stds, color=clr,
               label=cn, capsize=4, edgecolor='grey', lw=0.5, alpha=0.9,
               error_kw=dict(lw=1.2))
    ax.set_xticks(x); ax.set_xticklabels(feat_names, fontsize=9.5)
    ax.set_ylabel('10-fold CV Balanced Accuracy (%)')
    ax.set_ylim(78, 93)
    ax.legend(framealpha=0.8, fontsize=9)
    ax.set_title('(a) Feature-set comparison\n(10-fold CV, same labeled sample)',
                 fontweight='bold')
    ax.text(0.98, 0.04,
            f'Wilcoxon $p={pw:.3f}$ (RF)',
            transform=ax.transAxes, ha='right', fontsize=8.5,
            color='#D6604D',
            bbox=dict(boxstyle='round,pad=0.3', fc='white',
                      ec='#D6604D', alpha=0.85))

    ax = axes[1]
    ax.barh(CN[::-1], delta_f1[::-1], color=CC[::-1],
            alpha=0.85, edgecolor='grey', lw=0.5)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel(r'$\Delta F_1$ (10-feat $-$ 7-feat)')
    ax.set_title('(b) Per-class $F_1$ change\n(test set, RF)',
                 fontweight='bold')
    for i, (v, cls) in enumerate(zip(delta_f1[::-1], CN[::-1])):
        ax.text(v + (0.002 if v >= 0 else -0.002), i,
                f'{v:+.3f}', va='center',
                ha='left' if v >= 0 else 'right', fontsize=9)
    plt.tight_layout(pad=0.8)
    savefig('fig4_feature_comparison.pdf')

    # ── Figure 5: confusion matrix + ROC ──────────────────────────────
    fig = plt.figure(figsize=(12, 5))
    gs  = gridspec.GridSpec(1, 5, figure=fig, wspace=0.08)
    ax_cm  = fig.add_subplot(gs[0, :2])
    ax_roc = fig.add_subplot(gs[0, 2:])

    cm = confusion_matrix(y_te, yp10, labels=[0,1,2,3]).astype(float)
    cm_n = cm / cm.sum(axis=1, keepdims=True)
    im = ax_cm.imshow(cm_n, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04, label='Fraction')
    for i in range(4):
        for j in range(4):
            col = 'white' if cm_n[i, j] > 0.55 else 'black'
            ax_cm.text(j, i, f'{int(cm[i,j])}\n({100*cm_n[i,j]:.0f}%)',
                       ha='center', va='center', fontsize=9, color=col)
    ax_cm.set_xticks(range(4)); ax_cm.set_yticks(range(4))
    ax_cm.set_xticklabels(CN); ax_cm.set_yticklabels(CN)
    ax_cm.set_xlabel('Predicted class'); ax_cm.set_ylabel('True class')
    ax_cm.set_title('(a) Confusion matrix\n(RF, 10-feat, test set)',
                    fontweight='bold')

    y_te_bin = label_binarize(y_te, classes=[0, 1, 2, 3])
    ls_list  = ['-', '--', '-.', ':']
    for cid, (cls, col, ls) in enumerate(zip(CN, CC, ls_list)):
        fpr, tpr, _ = roc_curve(y_te_bin[:, cid], ypr10[:, cid])
        auc_v = roc_auc_(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=col, ls=ls, lw=1.8,
                    label=f'{cls} (AUC={auc_v:.3f})')
    ax_roc.plot([0, 1], [0, 1], 'k:', lw=0.9, alpha=0.5)
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('(b) One-vs-rest ROC\n(10-feat RF, test set)',
                     fontweight='bold')
    ax_roc.legend(loc='lower right', framealpha=0.85)
    ax_roc.set_aspect('equal')
    plt.tight_layout(pad=0.8)
    savefig('fig5_cm_roc.pdf')

    # ── Figure 6: feature importance ──────────────────────────────────
    fi    = rf10.feature_importances_
    order = np.argsort(fi)
    # Color by feature type
    type_colors = {
        'Log_Energy_Flux': 'steelblue', 'Log_Unc_Flux': 'steelblue',
        'Log_Signif': 'steelblue',
        'LP_index1GeV': CC[0], 'LP_beta': CC[0], 'LP_SigCurv': CC[0],
        'Log_Var': CC[1], 'Frac_Variability': CC[1],
        'GLAT_abs': CC[2], 'Log_LP_EPeak_imp': CC[0],
    }
    clrs_fi = [type_colors[FEAT_10[i]] for i in order]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(range(10), fi[order], color=clrs_fi,
            edgecolor='white', lw=0.4, alpha=0.88)
    ax.set_yticks(range(10))
    ax.set_yticklabels([FEAT_LABELS[i] for i in order], fontsize=10)
    ax.set_xlabel('Mean decrease in Gini impurity')
    ax.set_title('RF feature importance (10-feature set)',
                 fontweight='bold')
    for i, v in enumerate(fi[order]):
        ax.text(v + 0.003, i, f'{100*v:.1f}%', va='center', fontsize=8.5)
    legend_els = [
        Patch(fc=CC[0],      label='Spectral'),
        Patch(fc=CC[1],      label='Variability'),
        Patch(fc=CC[2],      label='Positional'),
        Patch(fc='steelblue',label='Flux/Significance'),
    ]
    ax.legend(handles=legend_els, loc='lower right',
              fontsize=8.5, framealpha=0.8)
    ax.set_xlim(0, fi.max() * 1.25)
    plt.tight_layout()
    savefig('fig6_feature_importance.pdf')

    # ── Figure 7: sky map + probability distribution ──────────────────
    df_u_plot = df_u.copy()
    if 'IsOOD' not in df_u_plot.columns:
        df_u_plot['IsOOD'] = False
    hc = df_u_plot[(df_u_plot['Confidence'] >= 0.80) &
                   (~df_u_plot['IsOOD'].astype(bool))]

    fig = plt.figure(figsize=(12, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    ax_sky  = fig.add_subplot(gs[0, 0])
    ax_prob = fig.add_subplot(gs[0, 1])

    for cls, col in zip(CN, CC):
        m = (hc['PredClass'] == cls)
        if m.sum() > 0:
            ax_sky.scatter(hc['GLON'][m], hc['GLAT'][m],
                           c=col, s=12, alpha=0.7,
                           label=f'{cls} ($n={m.sum()}$)',
                           rasterized=True, linewidths=0, zorder=3)
    ax_sky.axhline(0, color='gray', lw=0.7, ls='--', alpha=0.6)
    ax_sky.set_xlabel('Galactic Longitude $l$ (°)')
    ax_sky.set_ylabel('Galactic Latitude $b$ (°)')
    ax_sky.set_xlim(0, 360); ax_sky.set_ylim(-90, 90)
    ax_sky.legend(markerscale=1.5, fontsize=8.5, framealpha=0.85)
    ax_sky.set_title(f'(a) High-confidence candidates\n'
                     f'($N={len(hc)}$, '
                     r'$P_{\rm max}\geq0.80$, non-OOD)',
                     fontweight='bold')

    for cls, col in zip(CN, CC):
        m = (hc['PredClass'] == cls)
        if m.sum() >= 5:
            ax_prob.hist(hc['Confidence'][m], bins=20, density=True,
                         alpha=0.65, color=col, edgecolor='none',
                         label=f'{cls} ($n={m.sum()}$)')
    ax_prob.axvline(0.80, color='black', lw=1.2, ls='--',
                    label='Threshold (0.80)')
    ax_prob.set_xlabel(r'$P_{\rm max}$ (classification confidence)')
    ax_prob.set_ylabel('Density')
    ax_prob.legend(fontsize=8.5, framealpha=0.85)
    ax_prob.set_title('(b) Confidence distribution\n'
                      '(high-confidence candidates)',
                      fontweight='bold')
    plt.tight_layout(pad=0.8)
    savefig('fig7_sky_probabilities.pdf')

    print(f"\nAll figures saved to ./{FIGDIR}/")


if __name__ == '__main__':
    make_all_figures()
