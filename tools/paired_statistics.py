#!/usr/bin/env python3
"""Task-cluster bootstrap and exact paired McNemar test, with strict episode matching."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

def load_csv(path):
    rows={}
    with open(path,newline='') as f:
        for row in csv.DictReader(f):
            key=(row['task_id'],row['episode'])
            if key in rows:raise ValueError('Duplicate task/episode: '+str(key))
            v=int(row['success'])
            if v not in (0,1):raise ValueError('success must be 0 or 1')
            rows[key]=v
    if not rows:raise ValueError('No evaluation rows')
    return rows

def statistics(a,b,resamples=100000,seed=0):
    if set(a)!=set(b):raise ValueError('Paired evaluations must have identical task/episode identities.')
    keys=sorted(a);tasks=sorted({k[0] for k in keys})
    task_diffs=[]
    for task in tasks:
        ks=[k for k in keys if k[0]==task]
        task_diffs.append(np.mean([b[k]-a[k] for k in ks]))
    diffs=np.asarray(task_diffs,dtype=float)
    rng=np.random.default_rng(seed);draws=[]
    for start in range(0,resamples,512):
        idx=rng.integers(0,len(diffs),size=(min(512,resamples-start),len(diffs)))
        draws.extend(diffs[idx].mean(axis=1))
    lo,hi=np.quantile(draws,[0.025,0.975])*100
    a_only=sum(a[k] and not b[k] for k in keys);b_only=sum(b[k] and not a[k] for k in keys)
    p=binomtest(b_only,a_only+b_only,p=0.5).pvalue if a_only+b_only else 1.0
    return dict(tasks=len(tasks),paired_episodes=len(keys),a_success_pct=100*np.mean(list(a.values())),
        b_success_pct=100*np.mean(list(b.values())),task_mean_delta_pp=float(diffs.mean()*100),
        bootstrap95_pp=[float(lo),float(hi)],bootstrap_resamples=resamples,seed=seed,
        a_only=int(a_only),b_only=int(b_only),mcnemar_two_sided_p=float(p))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--a',required=True);ap.add_argument('--b',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--resamples',type=int,default=100000)
    ap.add_argument('--seed',type=int,default=0)
    a=ap.parse_args()
    if a.resamples<=0:raise SystemExit('resamples must be positive')
    report=statistics(load_csv(a.a),load_csv(a.b),a.resamples,a.seed)
    Path(a.out).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
