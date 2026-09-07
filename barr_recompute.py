# -*- coding: utf-8 -*-
"""
barr_recompute.py
Independent recomputation of the Barr normalised implicit distance for every
participant and every shape-matching trial, from the logged (a, b, c, m) values
and the study's own target-mesh assets.

    F(x,y,z) = ( |x/a|^m + |y/b|^m + |z/c|^m )^(1/m) - 1              (Barr, 1981)
    score    = 100 x mean_i |F(p_i)|   over N points sampled on the target surface
               minimised over all 6 permutations of (a, b, c)

Inputs
    User_Study_Data_clean.xlsx           sheet 'Combined'  (logged a,b,c,m)
    OriginalSuperquadricMesh_1.fbx       T1 target, semi-axes (0.50, 0.60, 0.80) dm
    OriginalSuperquadricMesh_2.fbx       T2 target, semi-axes (0.34, 0.71, 0.96) dm
    rock files/desktop/rock.obj          T3 target, Desktop cohort  (V = 1.05548 dm^3)

Notes
  * rock.obj contains two objects; the 4-vertex ground plane is discarded.
  * The XR rock (Rock_05_05_04_world.fbx) carries a non-uniform transform that is
    not recoverable without an FBX-aware loader, so the XR T3 target is
    reconstructed from the Desktop mesh using the documented ground-truth
    relationship (identical a and b; c scaled by 0.3834/0.3854, giving V = 1.0501).
    Set XR_FROM_FBX = True if an FBX loader is available to use the asset directly.

Usage:  python barr_recompute.py
Writes: barr_recomputed.csv
"""
import os, itertools, numpy as np, pandas as pd, trimesh
from openpyxl import load_workbook
from openpyxl.utils import column_index_from_string as ci

BASE = r"./study_data"   # directory holding the study mesh assets and workbooks
N_SAMPLES, SEED = 10000, 42
V_DESKTOP, V_XR = 1.05548, 1.0501
C_RATIO_XR = 0.3834 / 0.3854
XR_FROM_FBX = False


# ── metric ───────────────────────────────────────────────────────────────────
def barr_F(pts, a, b, c, m):
    return (np.abs(pts[:, 0] / a) ** m +
            np.abs(pts[:, 1] / b) ** m +
            np.abs(pts[:, 2] / c) ** m) ** (1.0 / m) - 1.0


def barr_score(pts, a, b, c, m):
    best, bp = None, None
    for p in itertools.permutations((a, b, c)):
        v = 100.0 * np.mean(np.abs(barr_F(pts, p[0], p[1], p[2], m)))
        if best is None or v < best:
            best, bp = v, p
    return best, bp


# ── mesh helpers ─────────────────────────────────────────────────────────────
def _fbx(path):
    from fbx_reader import to_mesh
    V, F = to_mesh(path)
    return trimesh.Trimesh(vertices=V, faces=F, process=True)


def _scale_vol(mesh, v):
    m = mesh.copy(); m.apply_scale((v / abs(mesh.volume)) ** (1 / 3)); return m


def _pca(mesh):
    m = mesh.copy()
    V = m.vertices - m.vertices.mean(0)
    w, U = np.linalg.eigh(np.cov(V.T))
    U = U[:, np.argsort(w)[::-1]]
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    m.vertices = V @ U
    return m


def _sample(mesh, n=N_SAMPLES, seed=SEED):
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(pts, float)


def _rock_desktop():
    sc = trimesh.load(os.path.join(BASE, 'rock files', 'desktop', 'rock.obj'),
                      force='scene', process=False)
    rock = max(sc.geometry.values(), key=lambda g: len(g.vertices))   # drop ground plane
    return _pca(_scale_vol(rock, V_DESKTOP))


def build_targets():
    t = {}
    for tag, fn in (('T1', 'OriginalSuperquadricMesh_1.fbx'),
                    ('T2', 'OriginalSuperquadricMesh_2.fbx')):
        m = _fbx(os.path.join(BASE, fn)); m.apply_scale(1 / 100.0)   # mm -> dm
        t[tag] = _sample(m)
    rd = _rock_desktop()
    t[('T3', 'Desktop')] = _sample(rd)
    if XR_FROM_FBX:
        rx = _pca(_scale_vol(_fbx(os.path.join(BASE, 'rock files', 'xr',
                                               'Rock_05_05_04_world.fbx')), V_XR))
    else:
        rx = rd.copy()
        rx.vertices = rx.vertices * np.array([1.0, 1.0, C_RATIO_XR])
    t[('T3', 'XR')] = _sample(rx)
    return t


COLS = {'T1': ('R', 'S', 'T', 'U'), 'T2': ('AD', 'AE', 'AF', 'AG'), 'T3': ('AP', 'AQ', 'AR', 'AS')}


def main():
    tp = build_targets()
    ws = load_workbook(os.path.join(BASE, 'User_Study_Data_clean.xlsx'),
                       data_only=True)['Combined']
    rows = []
    for r in range(4, ws.max_row + 1):
        pid, cond = ws.cell(row=r, column=1).value, ws.cell(row=r, column=2).value
        if pid is None or cond is None:
            continue
        cond = str(cond).strip()
        rec = {'PID': pid, 'Condition': cond,
               'Expertise': str(ws.cell(row=r, column=3).value).strip()}
        for tag, cols in COLS.items():
            p = [ws.cell(row=r, column=ci(c)).value for c in cols]
            if all(isinstance(x, (int, float)) and x > 0 for x in p):
                s, perm = barr_score(tp[(tag, cond)] if tag == 'T3' else tp[tag],
                                     *[float(x) for x in p])
                rec[tag + '_meanF_pct'] = round(s, 3)
                rec[tag + '_best_perm'] = '(%.2f,%.2f,%.2f)' % perm
            else:
                rec[tag + '_meanF_pct'] = np.nan
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv('barr_recomputed.csv', index=False)
    return df


if __name__ == '__main__':
    df = main()
    print('recomputed %d participants -> barr_recomputed.csv\n' % len(df))
    print(df.groupby('Condition')[['T1_meanF_pct', 'T2_meanF_pct', 'T3_meanF_pct']]
          .median().round(2).to_string())
