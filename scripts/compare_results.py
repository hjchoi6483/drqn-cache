#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, os
import numpy as np

def load_rows(path):
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--a', required=True)
    p.add_argument('--b', required=True)
    p.add_argument('--key_cols', default='scenario,alpha,cache_size,seed,setting')
    p.add_argument('--metric', default='rl_hit')
    p.add_argument('--out', default=None)
    args=p.parse_args()
    keys=[k.strip() for k in args.key_cols.split(',') if k.strip()]
    ra=load_rows(args.a); rb=load_rows(args.b)
    ma={tuple(r[k] for k in keys): float(r[args.metric]) for r in ra}
    mb={tuple(r[k] for k in keys): float(r[args.metric]) for r in rb}
    common=sorted(set(ma).intersection(mb))
    if not common:
        raise ValueError('No matching keys between A and B')
    diffs=np.array([ma[k]-mb[k] for k in common], dtype=np.float64)
    out={
        'n_pairs': int(len(diffs)),
        'metric': args.metric,
        'mean_diff_a_minus_b': float(np.mean(diffs)),
        'std_diff_a_minus_b': float(np.std(diffs, ddof=1)) if len(diffs)>1 else 0.0,
        'win_count_a_gt_b': int(np.sum(diffs>0.0)),
        'win_rate_a_gt_b': float(np.mean(diffs>0.0)),
    }
    try:
        from scipy import stats
        out['paired_ttest_pvalue']=float(stats.ttest_rel(diffs, np.zeros_like(diffs)).pvalue)
        if np.any(diffs != 0.0):
            out['wilcoxon_pvalue']=float(stats.wilcoxon(diffs).pvalue)
    except Exception:
        out['paired_ttest_pvalue']='NA(scipy unavailable)'
        out['wilcoxon_pvalue']='NA(scipy unavailable)'
    out_path=args.out or os.path.join(os.path.dirname(args.a) or '.', 'summary_pairwise_vs_main.csv')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(out.keys())); w.writeheader(); w.writerow(out)
    print(out)
    print(f'Saved: {out_path}')

if __name__=='__main__':
    main()
