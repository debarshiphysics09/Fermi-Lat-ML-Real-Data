#!/usr/bin/env python3
"""
02_covariate_shift.py — Quantify covariate shift between labeled and unassociated sources.
Input:  4fgl_dr4_features.csv
Output: shift_results.csv, shift_summary.txt
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

FEAT_10=['Log_Energy_Flux','Log_Unc_Flux','Log_Signif','LP_index1GeV',
         'LP_beta','LP_SigCurv','Log_Var','GLAT_abs','Log_LP_EPeak_imp','Frac_Variability']

def run(features_csv='4fgl_dr4_features.csv', seed=42):
    df=pd.read_csv(features_csv)
    df['IsLabeled']=df['IsLabeled'].astype(bool)
    df['IsUnassoc']=df['IsUnassoc'].astype(bool)
    df_lab=df[df.IsLabeled].copy(); df_u=df[df.IsUnassoc].copy()
    meds=df_lab[FEAT_10].median()
    df_lab[FEAT_10]=df_lab[FEAT_10].fillna(meds)
    df_u[FEAT_10]=df_u[FEAT_10].fillna(meds)
    print(f"Labeled: {len(df_lab)}   Unassociated: {len(df_u)}\n")
    print(f"{'Feature':<24} {'W1':>10} {'KS':>8} {'p-value':>14}"); print("-"*60)
    rows=[]
    for feat in FEAT_10:
        lv,uv=df_lab[feat].values,df_u[feat].values
        wd=wasserstein_distance(lv,uv); ks,p=ks_2samp(lv,uv)
        rows.append({'Feature':feat,'W1':wd,'KS':ks,'p_value':p})
        print(f"  {feat:<22} {wd:10.3f} {ks:8.3f} {p:14.3e}")
    pd.DataFrame(rows).to_csv('shift_results.csv',index=False)
    # Global shift classifier
    X=np.vstack([df_lab[FEAT_10].values,df_u[FEAT_10].values])
    y=np.array([0]*len(df_lab)+[1]*len(df_u))
    sc=StandardScaler().fit(X); Xsc=sc.transform(X)
    clf=RandomForestClassifier(n_estimators=200,min_samples_leaf=5,n_jobs=-1,random_state=seed)
    clf.fit(Xsc,y)
    shift_auc=roc_auc_score(y,clf.predict_proba(Xsc)[:,1])
    ts_p5=df_lab['Log_Signif'].quantile(0.05)
    n_ood=(df_u['Log_Signif']<ts_p5).sum()
    print(f"\nShift classifier AUC = {shift_auc:.3f}")
    print(f"OOD threshold (Log_Signif<{ts_p5:.3f}): {n_ood}/{len(df_u)} = {100*n_ood/len(df_u):.1f}%")
    with open('shift_summary.txt','w') as f:
        f.write(f"shift_auc={shift_auc:.4f}\nN_labeled={len(df_lab)}\nN_unassoc={len(df_u)}\n")
        f.write(f"ood_threshold={ts_p5:.4f}\nN_ood={n_ood}\n")
    print("Saved: shift_results.csv, shift_summary.txt")

if __name__=='__main__': run()
