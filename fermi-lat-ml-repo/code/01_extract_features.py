#!/usr/bin/env python3
"""
01_extract_features.py — Extract and engineer features from 4FGL-DR4 FITS file.
Input:  gll_psc_v35.fit
Output: 4fgl_dr4_features.csv
"""
import numpy as np, pandas as pd, struct, os, sys

BLOCK = 2880

def read_header(data, start):
    cards, pos = [], start
    while True:
        card = data[pos:pos+80].decode('ascii', errors='replace')
        cards.append(card); pos += 80
        if card.startswith('END'): break
    end = start + int(np.ceil((pos-start)/BLOCK))*BLOCK
    return cards, end

def parse_header_dict(cards):
    d = {}
    for card in cards:
        if card.startswith(('COMMENT','HISTORY','END')) or '=' not in card: continue
        key, rest = card.split('=',1)
        d[key.strip()] = rest.split('/')[0].strip().strip("'").strip()
    return d

def parse_tform(tform):
    tform=tform.strip(); i=0
    while i<len(tform) and tform[i].isdigit(): i+=1
    return (int(tform[:i]) if i>0 else 1), tform[i:].strip()

TYPE_MAP = {'I':('>h',2),'J':('>i',4),'K':('>q',8),'E':('>f',4),
            'D':('>d',8),'A':('s',1),'B':('B',1),'L':('c',1)}

def read_bintable(fits_path, ext=1, wanted=None):
    with open(fits_path,'rb') as f: data=f.read()
    _, pos = read_header(data, 0)
    cur=0
    while True:
        cards, hdr_end = read_header(data, pos)
        hdr=parse_header_dict(cards)
        cur+=1
        n1=int(hdr.get('NAXIS1',0)); n2=int(hdr.get('NAXIS2',0))
        tf=int(hdr.get('TFIELDS',0))
        dblocks=int(np.ceil(n1*n2/BLOCK)); ds=hdr_end
        if cur==ext:
            byte_off=0; col_defs=[]
            for c in range(1,tf+1):
                ttype=hdr.get(f'TTYPE{c}',f'COL{c}')
                tform=hdr.get(f'TFORM{c}','E')
                rep,tc=parse_tform(tform)
                _,unit=TYPE_MAP.get(tc,('>f',4))
                nb=rep*unit if tc!='A' else rep
                col_defs.append({'name':ttype,'rep':rep,'type':tc,'off':byte_off,'nb':nb})
                byte_off+=nb
            extract=[c for c in col_defs if (not wanted or c['name'] in wanted)]
            result={}
            for col in extract:
                tc,off,nb,rep,name=col['type'],col['off'],col['nb'],col['rep'],col['name']
                if tc=='A':
                    vals=[]
                    for row in range(n2):
                        raw=data[ds+row*n1+off:ds+row*n1+off+nb]
                        vals.append(raw.decode('ascii',errors='replace').strip())
                    result[name]=vals
                else:
                    fmt,unit=TYPE_MAP.get(tc,('>f',4))
                    if rep==1:
                        arr=np.empty(n2,dtype=np.float64)
                        for row in range(n2):
                            raw=data[ds+row*n1+off:ds+row*n1+off+unit]
                            arr[row]=struct.unpack(fmt,raw)[0]
                        result[name]=arr
                    else:
                        vals=[]
                        for row in range(n2):
                            raw=data[ds+row*n1+off:ds+row*n1+off+nb]
                            vals.append(np.array(struct.unpack(f'>{rep}{fmt[-1]}',raw)))
                        result[name]=vals
            return result, n2
        pos=hdr_end+dblocks*BLOCK
        if cur>20: raise RuntimeError("Extension not found")

def extract_features(fits_path='gll_psc_v35.fit', out='4fgl_dr4_features.csv'):
    if not os.path.exists(fits_path):
        print(f"ERROR: {fits_path} not found."); sys.exit(1)
    print(f"Reading {fits_path}...")
    wanted=['Source_Name','RAJ2000','DEJ2000','GLON','GLAT','Signif_Avg',
            'Pivot_Energy','Energy_Flux100','Unc_Energy_Flux100','SpectrumType',
            'PL_Index','Unc_PL_Index','LP_Index','LP_beta','LP_SigCurv','LP_EPeak',
            'Variability_Index','Frac_Variability','CLASS1','CLASS2','ASSOC1']
    cols,nrows=read_bintable(fits_path,1,wanted)
    df=pd.DataFrame(cols)
    for c in ['CLASS1','CLASS2','ASSOC1','SpectrumType','Source_Name']:
        df[c]=df[c].astype(str).fillna('').str.strip()
    df['CLASS1_clean']=df['CLASS1'].str.lower()
    df['Variability_Index']=pd.to_numeric(df['Variability_Index'],errors='coerce').replace([np.inf,-np.inf],np.nan)
    df['Frac_Variability']=pd.to_numeric(df['Frac_Variability'],errors='coerce').replace([np.inf,-np.inf],np.nan)
    # Derived features
    df['Log_Energy_Flux']=np.log10(pd.to_numeric(df['Energy_Flux100'],errors='coerce').clip(lower=1e-20))
    df['Log_Unc_Flux']=np.log10(pd.to_numeric(df['Unc_Energy_Flux100'],errors='coerce').clip(lower=1e-20))
    df['Log_Signif']=np.log10(pd.to_numeric(df['Signif_Avg'],errors='coerce').clip(lower=1e-5))
    df['LP_index1GeV']=(pd.to_numeric(df['LP_Index'],errors='coerce')+
        2*pd.to_numeric(df['LP_beta'],errors='coerce')*
        np.log(1000.0/pd.to_numeric(df['Pivot_Energy'],errors='coerce').clip(lower=1)))
    df['LP_beta']=pd.to_numeric(df['LP_beta'],errors='coerce')
    df['LP_SigCurv']=pd.to_numeric(df['LP_SigCurv'],errors='coerce')
    df['Log_Var']=np.log10(df['Variability_Index'].clip(lower=1e-5))
    df['GLAT_abs']=pd.to_numeric(df['GLAT'],errors='coerce').abs()
    ep_raw=pd.to_numeric(df['LP_EPeak'],errors='coerce').clip(lower=1.0)
    med_ep=np.log10(ep_raw.median())
    df['Log_LP_EPeak_imp']=np.log10(ep_raw).fillna(med_ep)
    df['Frac_Variability']=df['Frac_Variability'].fillna(0.0)
    cls_map={'bll':0,'BLL':0,'fsrq':1,'FSRQ':1,'psr':2,'PSR':2,'msp':3,'MSP':3}
    df['ClassID']=df['CLASS1'].map(cls_map)
    df['IsLabeled']=df['ClassID'].notna()
    df['IsUnassoc']=(df['CLASS1']=='')
    df['IsBCU']=(df['CLASS1_clean']=='bcu')
    df.to_csv(out,index=False)
    print(f"Saved {out} ({len(df)} rows)")
    lab=df[df.IsLabeled]; print(f"  Labeled: {len(lab)}  Unassoc: {df.IsUnassoc.sum()}  BCU: {df.IsBCU.sum()}")
    for cls,cid in [('BLL',0),('FSRQ',1),('PSR',2),('MSP',3)]:
        print(f"    {cls}: {(lab.ClassID==cid).sum()}")

if __name__=='__main__': extract_features()
