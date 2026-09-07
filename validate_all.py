# -*- coding: utf-8 -*-
"""Fidelity report: recomputed vs published, plus metric validation on the
   five known-parameter ValidateMesh assets."""
import os, numpy as np, pandas as pd, trimesh
from scipy import stats
from barr_recompute import (main, barr_score, _fbx, _sample, BASE)

df = main()
print('=' * 76)
print('A. RECOMPUTED vs PUBLISHED  (per participant)')
print('=' * 76)
t12 = pd.read_excel(os.path.join(BASE, 'trial12_analysis', 'trial12_analysis_results.xlsx'),
                    sheet_name='Per_Participant')
rock = pd.read_excel(os.path.join(BASE, 'barr_analysis', 'barr_analysis_results.xlsx'),
                     sheet_name='Barr_Analysis')
rock.columns = [c.strip() for c in rock.columns]
pub = {'T1': dict(zip(t12.PID, t12['T1_mean|F|%'])),
       'T2': dict(zip(t12.PID, t12['T2_mean|F|%'])),
       'T3': dict(zip(rock.PID, rock['mean_abs_F_pct']))}


def rb(a, b):
    u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    return u, 1 - 2 * u / (len(a) * len(b)), p


PUB_R = {'T1': 0.56, 'T2': 0.42, 'T3': 0.35}
for tag in ('T1', 'T2', 'T3'):
    mine = df[tag + '_meanF_pct'].values
    theirs = np.array([pub[tag].get(p, np.nan) for p in df.PID])
    ok = ~(np.isnan(mine) | np.isnan(theirs))
    d = mine[ok] - theirs[ok]
    print('\n%s   n=%d   corr=%.4f   mean diff=%+.3f pp   sd=%.3f'
          % (tag, ok.sum(), np.corrcoef(mine[ok], theirs[ok])[0, 1], d.mean(), d.std()))
    xr = df.Condition.str.upper().values == 'XR'
    a, b = mine[xr & ok], mine[(~xr) & ok]
    u, r, p = rb(a, b)
    ap, bp = theirs[xr & ok], theirs[(~xr) & ok]
    u2, r2, p2 = rb(ap, bp)
    print('   recomputed : Mdn %.2f vs %.2f   U=%.1f  r=%.3f  p=%.5f' % (np.median(a), np.median(b), u, r, p))
    print('   published  : Mdn %.2f vs %.2f   U=%.1f  r=%.3f  p=%.5f   (paper r=%.2f)'
          % (np.median(ap), np.median(bp), u2, r2, p2, PUB_R[tag]))

print()
print('=' * 76)
print('B. METRIC VALIDATION on ValidateMesh_1..5 (known parameters)')
print('=' * 76)
SHEET = {1: (0.345, 0.274, 0.877, 3.456), 2: (1.237, 2.543, 1.987, 2.315),
         3: (0.845, 0.586, 0.684, 1.987), 4: (1.453, 1.224, 0.245, 4.215),
         5: (0.668, 0.825, 0.745, 0.314)}
print('%-6s %-28s %10s %10s %10s' % ('mesh', 'true (a,b,c,m)', 'score@true', '+5% on a', '+10% on m'))
for i in range(1, 6):
    m = _fbx(os.path.join(BASE, 'ValidateMesh_%d.fbx' % i))
    m.apply_scale(1 / 100.0)
    pts = _sample(m)
    a, b, c, mm = SHEET[i]
    s0, _ = barr_score(pts, a, b, c, mm)
    s1, _ = barr_score(pts, a * 1.05, b, c, mm)
    s2, _ = barr_score(pts, a, b, c, mm * 1.10)
    print('%-6d %-28s %9.3f%% %9.3f%% %9.3f%%'
          % (i, '(%.3f,%.3f,%.3f,%.3f)' % (a, b, c, mm), s0, s1, s2))
