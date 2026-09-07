# -*- coding: utf-8 -*-
"""
interaction_analysis.py
Condition x Expertise interaction tests for the TVCG submission.

Self-contained: reads only `user_study_data_deidentified.csv` from this directory.

Reproduces the values reported in Sec. 4.1.2:
    T1 Barr shape-form error : F(1,60) = 0.01, p = .93
    T2 Barr shape-form error : F(1,60) = 1.00, p = .32
    T3 absolute volume error : F(1,60) = 0.05, p = .82
and the Condition main effect on Barr shape-form error (significant for all three targets),
and the expertise main effects reported in the same subsection.

Method: aligned rank transform (Wobbrock et al. 2011) for a 2x2 between-subjects design,
followed by a factorial ANOVA on the aligned ranks. ART is used rather than a raw parametric
ANOVA because the error distributions are right-skewed -- the same reason Mann-Whitney is
used for the primary contrasts.

Usage:  python interaction_analysis.py
Requires: numpy, pandas, scipy
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, 'user_study_data_deidentified.csv')


# ── aligned rank transform ───────────────────────────────────────────────────
def art_anova(y, A, B, effect='AB'):
    """ART for a 2x2 between-subjects design.

    y : response,  A, B : 0/1 factor codes,  effect : 'A', 'B' or 'AB'.
    Returns (F, p, df_error).
    """
    y = np.asarray(y, float)
    cell, cnt = {}, {}
    for a in (0, 1):
        for b in (0, 1):
            m = (A == a) & (B == b)
            cell[(a, b)], cnt[(a, b)] = y[m].mean(), int(m.sum())

    # estimated marginal (unweighted) means -- the design is unbalanced
    gm = np.mean(list(cell.values()))
    rowm = {a: np.mean([cell[(a, 0)], cell[(a, 1)]]) for a in (0, 1)}
    colm = {b: np.mean([cell[(0, b)], cell[(1, b)]]) for b in (0, 1)}

    cm = np.array([cell[(a, b)] for a, b in zip(A, B)])
    resid = y - cm                                   # strip all effects
    if effect == 'AB':
        eff = np.array([cell[(a, b)] - rowm[a] - colm[b] + gm for a, b in zip(A, B)])
    elif effect == 'A':
        eff = np.array([rowm[a] - gm for a in A])
    else:
        eff = np.array([colm[b] - gm for b in B])

    r = stats.rankdata(resid + eff)                  # align, then rank

    n = len(r)
    g = r.mean()
    rm = {a: r[A == a].mean() for a in (0, 1)}
    cl = {b: r[B == b].mean() for b in (0, 1)}
    cr = {(a, b): r[(A == a) & (B == b)].mean() for a in (0, 1) for b in (0, 1)}

    ss = {
        'A':  sum(cnt[k] * (rm[k[0]] - g) ** 2 for k in cr),
        'B':  sum(cnt[k] * (cl[k[1]] - g) ** 2 for k in cr),
        'AB': sum(cnt[k] * (cr[k] - rm[k[0]] - cl[k[1]] + g) ** 2 for k in cr),
    }[effect]
    sse = float(sum((r[i] - cr[(A[i], B[i])]) ** 2 for i in range(n)))
    dfe = n - 4
    F = (ss / 1.0) / (sse / dfe)
    return F, float(1 - stats.f.cdf(F, 1, dfe)), dfe


def load():
    df = pd.read_csv(CSV)
    df['C'] = df['Condition'].astype(str).str.strip().str.upper().str[:2].map(
        lambda s: 'XR' if s == 'XR' else 'Desktop')
    df['E'] = df['Expert/Novice (E/N)'].astype(str).str.strip().str.upper().str[0]
    out = []
    for label, col, absolute in [
            ('T1 Barr shape-form error', 'T1_barr_meanF_pct', False),
            ('T2 Barr shape-form error', 'T2_barr_meanF_pct', False),
            ('T3 absolute volume error', 'T3_vol_err_signed_pct', True)]:
        d = df[['C', 'E', col]].dropna().rename(columns={col: 'y'})
        if absolute:
            d['y'] = d.y.abs()
        out.append((label, d))
    return df, out


if __name__ == '__main__':
    df, sets = load()
    print('n = %d   cells: %s\n' % (len(df), df.groupby(['C', 'E']).size().to_dict()))
    print('Condition x Expertise -- aligned rank transform, 2x2 between-subjects\n')
    print('%-26s %24s %26s' % ('dependent variable', 'interaction', 'Condition main effect'))
    print('-' * 78)
    for label, d in sets:
        A = (d.C.values == 'XR').astype(int)
        B = (d.E.values == 'E').astype(int)
        Fi, pi, dfe = art_anova(d.y.values, A, B, 'AB')
        Fm, pm, _ = art_anova(d.y.values, A, B, 'A')
        print('%-26s F(1,%d)=%6.3f p=%.3f    F(1,%d)=%7.3f p=%.4f'
              % (label, dfe, Fi, pi, dfe, Fm, pm))

    # expertise main effects reported in Sec. 4.1.2
    print('\nExpert vs Novice (Mann-Whitney, pooled across condition)')
    print('-' * 78)
    for label, col, absolute in [
            ('T1 Barr shape-form', 'T1_barr_meanF_pct', False),
            ('T2 Barr shape-form', 'T2_barr_meanF_pct', False),
            ('T3 Barr shape-form', 'T3_barr_meanF_pct', False),
            ('T1 absolute volume', 'T1_vol_err_signed_pct', True),
            ('T2 absolute volume', 'T2_vol_err_signed_pct', True),
            ('T3 absolute volume', 'T3_vol_err_signed_pct', True)]:
        d = df[['E', col]].dropna()
        y = d[col].abs() if absolute else d[col]
        a, b = y[d.E == 'E'].values, y[d.E == 'N'].values
        U, p = stats.mannwhitneyu(a, b, alternative='two-sided')
        r = 1 - 2 * U / (len(a) * len(b))
        print('  %-20s U=%6.1f  r=%+0.3f  p=%.4f%s'
              % (label, U, r, p, '  *' if p < .05 else ''))

    print('\nCell sizes: XR-Expert 15, XR-Novice 17, Desktop-Expert 14, Desktop-Novice 18.')
