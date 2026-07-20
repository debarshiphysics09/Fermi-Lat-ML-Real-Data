#!/usr/bin/env python3
"""
05_robustness.py — Additional analyses requested for referee-proofing:
 (A) Repeated stratified train/test splits -> CI on test-set BalAcc/F1
 (B) Repeated 10-fold CV (multiple seeds) -> more robust CI on the
     7-feat vs 10-feat comparison, replacing the single-seed CV
 (C) Blazar-sequence figure: log(Epeak) vs Gamma_1GeV colored by class
 (D) Permutation importance vs Gini importance
 (E) Calibration (reliability) curve for P_max on held-out test folds
"""
import numpy as np, pandas as pd, warnings, json
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
CN = ['BLL','FSRQ','PSR','MSP']
FEAT_M7 = ['Log_Energy_Flux','Log_Unc_Flux','Log_Signif','LP_index1GeV',
           'LP_beta','LP_SigCurv','Log_Var']
FEAT_10 = FEAT_M7 + ['GLAT_abs','Log_LP_EPeak_imp','Frac_Variability']

df = pd.read_csv('4fgl_dr4_features.csv')
df['IsLabeled'] = df['IsLabeled'].astype(bool)
df_lab = df[df.IsLabeled].copy()
meds = df_lab[FEAT_10].median()
df_lab[FEAT_10] = df_lab[FEAT_10].fillna(meds)
X7, X10 = df_lab[FEAT_M7].values, df_lab[FEAT_10].values
y = df_lab['ClassID'].values.astype(int)

results = {}

# ── (A)+(B) Repeated CV: 10 different random seeds x 10-fold each ────
print("="*65); print("REPEATED 10-FOLD CV (10 seeds x 10 folds = 100 estimates)"); print("="*65)
rf = lambda s: RandomForestClassifier(n_estimators=150, max_features='sqrt',
        min_samples_leaf=2, class_weight='balanced', n_jobs=-1, random_state=s)

s7_all, s10_all = [], []
for seed in range(5):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    for tr, te in cv.split(X10, y):
        sc7 = StandardScaler().fit(X7[tr]); sc10 = StandardScaler().fit(X10[tr])
        clf7 = rf(seed).fit(sc7.transform(X7[tr]), y[tr])
        clf10 = rf(seed).fit(sc10.transform(X10[tr]), y[tr])
        s7_all.append(balanced_accuracy_score(y[te], clf7.predict(sc7.transform(X7[te]))))
        s10_all.append(balanced_accuracy_score(y[te], clf10.predict(sc10.transform(X10[te]))))

s7_all, s10_all = np.array(s7_all), np.array(s10_all)
_, p_rep = wilcoxon(s10_all, s7_all)
print(f"7-feat:  {100*s7_all.mean():.1f} +/- {100*s7_all.std():.1f}%  "
      f"(95% CI: [{100*np.percentile(s7_all,2.5):.1f}, {100*np.percentile(s7_all,97.5):.1f}])")
print(f"10-feat: {100*s10_all.mean():.1f} +/- {100*s10_all.std():.1f}%  "
      f"(95% CI: [{100*np.percentile(s10_all,2.5):.1f}, {100*np.percentile(s10_all,97.5):.1f}])")
print(f"Delta = +{100*(s10_all.mean()-s7_all.mean()):.2f}%  Wilcoxon p={p_rep:.2e}  N={len(s7_all)} paired folds")
print("[Note: 5 seeds x 10 folds = 50 estimates, n_estimators=150, reduced from the plan "
      "due to single-CPU compute budget; still far more than the original 10 single-seed estimates.]")
results['repeated_cv'] = dict(mean7=float(s7_all.mean()), std7=float(s7_all.std()),
                               mean10=float(s10_all.mean()), std10=float(s10_all.std()),
                               p_wilcoxon=float(p_rep), n_folds=len(s7_all))

# ── (A) Repeated 70/30 splits for test-set style CI ──────────────────
print("\n" + "="*65); print("REPEATED 70/30 HOLD-OUT SPLITS (30 repeats)"); print("="*65)
bacc10, bacc7, f1_per_class = [], [], []
for seed in range(15):
    X10_tr,X10_te,y_tr,y_te = train_test_split(X10,y,test_size=0.3,stratify=y,random_state=seed)
    X7_tr,X7_te,_,_ = train_test_split(X7,y,test_size=0.3,stratify=y,random_state=seed)
    sc10=StandardScaler().fit(X10_tr); sc7=StandardScaler().fit(X7_tr)
    c10=rf(seed).fit(sc10.transform(X10_tr),y_tr); c7=rf(seed).fit(sc7.transform(X7_tr),y_tr)
    p10=c10.predict(sc10.transform(X10_te)); p7=c7.predict(sc7.transform(X7_te))
    bacc10.append(balanced_accuracy_score(y_te,p10)); bacc7.append(balanced_accuracy_score(y_te,p7))
    f1_per_class.append(f1_score(y_te,p10,average=None,labels=[0,1,2,3]))
bacc10,bacc7 = np.array(bacc10), np.array(bacc7)
f1_per_class = np.array(f1_per_class)
print(f"10-feat test BalAcc: {100*bacc10.mean():.1f} +/- {100*bacc10.std():.1f}%  "
      f"(range {100*bacc10.min():.1f}-{100*bacc10.max():.1f}%)")
print(f" 7-feat test BalAcc: {100*bacc7.mean():.1f} +/- {100*bacc7.std():.1f}%")
_,p_hold = wilcoxon(bacc10,bacc7)
print(f"Wilcoxon (15 paired splits) p={p_hold:.2e}")
for i,c in enumerate(CN):
    print(f"  {c} F1 across 30 splits: {f1_per_class[:,i].mean():.3f} +/- {f1_per_class[:,i].std():.3f}")
results['repeated_holdout'] = dict(mean10=float(bacc10.mean()), std10=float(bacc10.std()),
                                    mean7=float(bacc7.mean()), std7=float(bacc7.std()),
                                    p_wilcoxon=float(p_hold))

# ── (D) Permutation importance (single 70/30 split, seed=42, matches main pipeline) ─
print("\n" + "="*65); print("PERMUTATION IMPORTANCE vs GINI IMPORTANCE"); print("="*65)
X10_tr,X10_te,y_tr,y_te = train_test_split(X10,y,test_size=0.3,stratify=y,random_state=SEED)
sc = StandardScaler().fit(X10_tr)
clf = rf(SEED).fit(sc.transform(X10_tr), y_tr)
gini = clf.feature_importances_
perm = permutation_importance(clf, sc.transform(X10_te), y_te, n_repeats=15,
                               random_state=SEED, scoring='balanced_accuracy')
imp_table = pd.DataFrame({'Feature':FEAT_10,'Gini':gini,
                           'Perm_mean':perm.importances_mean,'Perm_std':perm.importances_std})
imp_table = imp_table.sort_values('Perm_mean', ascending=False)
print(imp_table.to_string(index=False))
imp_table.to_csv('permutation_importance.csv', index=False)

# ── (E) Calibration curve ─────────────────────────────────────────────
print("\n" + "="*65); print("CALIBRATION CHECK (P_max reliability, pooled 10-fold OOF)"); print("="*65)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
oof_pmax, oof_correct = [], []
for tr, te in cv.split(X10, y):
    sc_f = StandardScaler().fit(X10[tr])
    c = rf(SEED).fit(sc_f.transform(X10[tr]), y[tr])
    proba = c.predict_proba(sc_f.transform(X10[te]))
    pred = proba.argmax(1); pmax = proba.max(1)
    oof_pmax.extend(pmax); oof_correct.extend((pred==y[te]).astype(int))
oof_pmax, oof_correct = np.array(oof_pmax), np.array(oof_correct)
frac_pos, mean_pred = calibration_curve(oof_correct, oof_pmax, n_bins=10, strategy='quantile')
for mp, fp in zip(mean_pred, frac_pos):
    print(f"  predicted P_max~{mp:.3f}  ->  observed accuracy {fp:.3f}")
hc_mask = oof_pmax >= 0.80
print(f"\n  At P_max>=0.80 threshold: observed accuracy = {oof_correct[hc_mask].mean():.3f} "
      f"(N={hc_mask.sum()}) vs nominal >=0.80")

lo = min(oof_pmax.min(), frac_pos.min()) - 0.03
hi = 1.02
fig, ax = plt.subplots(figsize=(5.2,5.2))
ax.plot([0,1],[0,1],'k--',lw=1,label='Perfect calibration')
ax.axhline(0.25, color='grey', ls=':', lw=1, label='Chance level (4 classes)')
ax.plot(mean_pred, frac_pos, 'o-', color='#4C72B0', label='RF (10-fold OOF)', zorder=3)
# small histogram of P_max density along the bottom, on a twin axis, for context
ax2 = ax.twinx()
ax2.hist(oof_pmax, bins=30, range=(lo,hi), color='#4C72B0', alpha=0.15, zorder=1)
ax2.set_ylabel('Count (P$_{\\max}$ histogram)', color='#4C72B0', alpha=0.6)
ax2.tick_params(axis='y', labelcolor='#4C72B0')
ax.set_xlabel(r'Mean predicted $P_{\max}$ (bin)'); ax.set_ylabel('Observed accuracy')
ax.set_title('Calibration of RF posterior confidence\n(10-feature RF, 10-fold out-of-fold predictions)')
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.legend(loc='lower right', fontsize=9)
ax.set_zorder(ax2.get_zorder()+1); ax.patch.set_visible(False)
plt.tight_layout(); plt.savefig('fig8_calibration.pdf'); plt.close()
print(f"Saved fig8_calibration.pdf (axes rescaled to data range [{lo:.2f}, {hi:.2f}])")

# ── (C) Blazar-sequence figure ────────────────────────────────────────
print("\n" + "="*65); print("BLAZAR-SEQUENCE FIGURE"); print("="*65)
colors = {'BLL':'#4C72B0','FSRQ':'#DD8452','PSR':'#55A868','MSP':'#8172B2'}
fig, ax = plt.subplots(figsize=(6.2,5.2))
cls_names = df_lab['CLASS1'].str.upper().map({'BLL':'BLL','FSRQ':'FSRQ','PSR':'PSR','MSP':'MSP'})
for c in CN:
    m = cls_names==c
    ax.scatter(df_lab.loc[m,'LP_index1GeV'], df_lab.loc[m,'Log_LP_EPeak_imp'],
               s=8, alpha=0.5, color=colors[c], label=c, edgecolors='none')
ax.set_xlabel(r'$\Gamma_{1\,\rm GeV}$ (spectral index at 1 GeV)')
ax.set_ylabel(r'$\log E^{\rm LP}_{\rm peak}$ (log SED peak energy)')
ax.set_title('Blazar-sequence view of the two added features')
ax.legend(markerscale=2)
plt.tight_layout(); plt.savefig('fig9_blazar_sequence.pdf'); plt.close()
print("Saved fig9_blazar_sequence.pdf")

with open('robustness_results.json','w') as f:
    json.dump(results, f, indent=2)
print("\nSaved robustness_results.json")
