"""self_patches.py (9 Nov 2026): cut automatic "good region" patches from a spiral fit's own winding meshes, for the
self-patch A/B of PREREG_selfpatch.md (the team's ask: "automatically crop 'good' regions of the spiral fit, and use
these as surface patch inputs to a subsequent run").

Inputs are two fits of the same band that differ only in the random seed (arm A seed 1 and arm A seed 2 of
PREREG_patches_ab.md). No human patch, held-out or training, is read.

A node of a seed-1 winding is GOOD when all of these hold:
  1. seed agreement: the nearest seed-2 winding surface is within --agree vox (exact point-to-triangle distance)
     AND carries the same winding index;
  2. single sheet: no node of another seed-1 winding lies within 25 vox sideways and closer than 0.03 mm along the
     normal (skuzbucket1's duplicate-sheet rule, as in score_ab.py);
  3. the node is valid and inside the band.
Tiles: each winding grid (rows ~ z, columns ~ angle) is cut into column blocks of --cols columns (full height). In
--mode select a tile is kept when >= --min-frac of its valid nodes are GOOD; non-GOOD nodes are set to -1, then only
the largest 4-connected GOOD component is kept; kept tiles need >= --min-nodes nodes.
In --mode random (the control), the same number of tiles is drawn at random (seeded) from all tiles with >= --min-nodes
valid nodes, ignoring GOOD, and kept whole.
Each tile is written as <out>/<prefix><winding>_c<col0>.tifxyz (x/y/z.tif float32, meta.json) and a summary JSON.

usage: self_patches.py --fit <A1 meshes> --other <A2 meshes> --out <dir> --prefix slf_ --mode select|random
       [--agree 2 --cols 60 --min-frac 0.6 --min-nodes 800 --seed 0 --summary <json>]
"""
import argparse
import json
import os
import sys

import numpy as np
import tifffile
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from score_ab import load_windings, mesh_triangles, nearest_winding_surface, VOXEL_UM  # noqa: E402

VOLUME_ID = '20250821151723'
VOLUME_SHAPE_ZYX = [16993, 7947, 7947]


def dup_mask(windings, um=VOXEL_UM, same_mm=0.03, lateral=25.0, k=12):
    """Per-node duplicate-sheet flag (True = another winding shares this sheet), same rule as score_ab.py."""
    P, N, owner = [], [], []
    masks = []
    for i, (w, name, zyx, valid) in enumerate(windings):
        Z, Y, X = zyx[..., 0], zyx[..., 1], zyx[..., 2]
        ok = (X > 0) & (Y > 0) & (Z > 0)
        good = np.zeros_like(ok)
        if min(X.shape) >= 3:
            Xi, Xj = np.gradient(X); Yi, Yj = np.gradient(Y); Zi, Zj = np.gradient(Z)
            nx, ny, nz = Yi * Zj - Zi * Yj, Zi * Xj - Xi * Zj, Xi * Yj - Yi * Xj
            nn = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
            good = ok & (nn > 0) & np.isfinite(nn)
            good[1:-1, 1:-1] &= ok[:-2, 1:-1] & ok[2:, 1:-1] & ok[1:-1, :-2] & ok[1:-1, 2:]
            good[[0, -1], :] = False
            good[:, [0, -1]] = False
            P.append(np.stack([X[good], Y[good], Z[good]], 1))
            N.append(np.stack([nx[good], ny[good], nz[good]], 1) / nn[good][:, None])
            owner.append(np.full(int(good.sum()), i))
        masks.append(good)
    P, N, owner = np.concatenate(P), np.concatenate(N), np.concatenate(owner)
    same_vx = same_mm * 1000 / um
    dist, idx = cKDTree(P).query(P, k=k, distance_upper_bound=lateral)
    dup = np.zeros(len(P), bool)
    for c in range(1, k):
        j = idx[:, c]
        have = np.isfinite(dist[:, c])
        jj = np.where(have, j, 0)
        d = P[jj] - P
        perp = np.abs((d * N[jj]).sum(1))
        lat = np.sqrt(np.maximum((d * d).sum(1) - perp ** 2, 0))
        dup |= have & (owner[jj] != owner) & (perp < same_vx) & (lat < lateral)
    out, s = [], 0
    for i, good in enumerate(masks):
        m = np.ones(good.shape, bool)          # nodes without a normal (edges, holes) count as not single-sheet
        n = int(good.sum())
        m[good] = dup[s:s + n]
        s += n
        out.append(m)
    return out


def agreement(windings, other, agree, chunk=200000):
    mesh = mesh_triangles(other)
    out = []
    for w, name, zyx, valid in windings:
        pts = zyx[valid]
        dist = np.full(len(pts), np.inf); win = np.full(len(pts), -1)
        for a in range(0, len(pts), chunk):
            d, wi, _ = nearest_winding_surface(pts[a:a + chunk], mesh, max_dist=agree)
            dist[a:a + chunk], win[a:a + chunk] = d, wi
        g = np.zeros(valid.shape, bool)
        g[valid] = (dist <= agree) & (win == w)
        out.append((g, int(valid.sum()), int(((dist <= agree)).sum()), int(((dist <= agree) & (win == w)).sum())))
    return out


def write_tile(path, zyx, valid, uuid, note):
    os.makedirs(path, exist_ok=True)
    t = zyx.astype(np.float32).copy()
    t[~valid] = -1
    for k, c in enumerate('zyx'):
        tifffile.imwrite(os.path.join(path, f'{c}.tif'), t[..., k])
    v = zyx[valid]
    meta = dict(format='tifxyz', type='seg', uuid=uuid, scale=[0.05, 0.05],
                bbox=[[float(v[:, 2].min()), float(v[:, 1].min()), float(v[:, 0].min())],
                      [float(v[:, 2].max()), float(v[:, 1].max()), float(v[:, 0].max())]],
                voxel_size_um=VOXEL_UM, voxelsize=VOXEL_UM, scroll='PHerc0813', volume_id=VOLUME_ID,
                volume_shape_zyx=VOLUME_SHAPE_ZYX, source='self_patches.py', note=note)
    json.dump(meta, open(os.path.join(path, 'meta.json'), 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit', required=True); ap.add_argument('--other', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--prefix', required=True)
    ap.add_argument('--mode', choices=['select', 'random'], required=True)
    ap.add_argument('--agree', type=float, default=2.0); ap.add_argument('--cols', type=int, default=60)
    ap.add_argument('--min-frac', type=float, default=0.6); ap.add_argument('--min-nodes', type=int, default=800)
    ap.add_argument('--n-tiles', type=int, default=0, help='random mode: number of tiles (default: from --select-summary)')
    ap.add_argument('--select-summary', default=None); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--z0', type=float, default=9000); ap.add_argument('--z1', type=float, default=10000)
    ap.add_argument('--summary', required=True)
    a = ap.parse_args()

    W = load_windings(a.fit)
    W = [(w, n, zyx, valid & (zyx[..., 0] >= a.z0) & (zyx[..., 0] < a.z1)) for w, n, zyx, valid in W]
    summ = dict(fit=a.fit, other=a.other, mode=a.mode, prefix=a.prefix, agree_vox=a.agree, cols=a.cols,
                min_frac=a.min_frac, min_nodes=a.min_nodes, seed=a.seed, windings=len(W))
    tiles = []   # (winding, name, c0, mask)
    if a.mode == 'select':
        O = load_windings(a.other)
        AG = agreement(W, O, a.agree)
        DU = dup_mask(W)
        nv = sum(x[1] for x in AG)
        summ['nodes_valid'] = nv
        summ['share_near_other_seed'] = sum(x[2] for x in AG) / nv
        summ['share_same_winding_other_seed'] = sum(x[3] for x in AG) / nv
        good_all = 0
        for (w, name, zyx, valid), (g, *_), du in zip(W, AG, DU):
            good = valid & g & ~du
            good_all += int(good.sum())
            for c0 in range(0, zyx.shape[1], a.cols):
                sl = np.s_[:, c0:c0 + a.cols]
                v, gd = valid[sl], good[sl]
                if v.sum() == 0 or gd.sum() < a.min_frac * v.sum():
                    continue
                lab, n = ndimage.label(gd)
                if n == 0:
                    continue
                big = np.argmax(ndimage.sum(gd, lab, range(1, n + 1))) + 1
                m = lab == big
                if m.sum() >= a.min_nodes:
                    tiles.append((w, name, c0, m))
        summ['share_good'] = good_all / nv
    else:
        cand = []
        for w, name, zyx, valid in W:
            for c0 in range(0, zyx.shape[1], a.cols):
                v = valid[:, c0:c0 + a.cols]
                if v.sum() >= a.min_nodes:
                    cand.append((w, name, c0, v.copy()))
        n = a.n_tiles or json.load(open(a.select_summary))['tiles']
        rng = np.random.default_rng(a.seed)
        pick = sorted(rng.choice(len(cand), size=min(n, len(cand)), replace=False))
        tiles = [cand[i] for i in pick]
        summ['candidate_tiles'] = len(cand)
    os.makedirs(a.out, exist_ok=True)
    byname = {name: zyx for w, name, zyx, _ in W}
    nodes = 0
    rows = []
    for w, name, c0, m in tiles:
        zyx = byname[name][:, c0:c0 + a.cols]
        uuid = f'{a.prefix}w{w:03d}_c{c0:03d}'
        write_tile(os.path.join(a.out, uuid + '.tifxyz'), zyx, m, uuid,
                   f'{a.mode} tile of {name} columns {c0}-{c0 + a.cols} ({os.path.basename(a.fit.rstrip("/"))})')
        nodes += int(m.sum())
        rows.append(dict(uuid=uuid, winding=w, c0=c0, nodes=int(m.sum()),
                         z=[float(zyx[m][:, 0].min()), float(zyx[m][:, 0].max())]))
    step_um = 20 * VOXEL_UM   # nominal node spacing (~19-24 vox in the fit meshes)
    summ.update(tiles=len(tiles), nodes=nodes, area_cm2_nominal=nodes * (step_um * 1e-4) ** 2,
                windings_with_tiles=len({r['winding'] for r in rows}), per_tile=rows)
    json.dump(summ, open(a.summary, 'w'), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != 'per_tile'}, indent=1))


if __name__ == '__main__':
    main()
