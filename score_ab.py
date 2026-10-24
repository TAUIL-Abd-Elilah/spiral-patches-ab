"""Score the patches A/B of PREREG_patches_ab.md on CPU from each arm's exported winding meshes (tifxyz, the raw
`wNNN_<tag>` windings; `_spliced` copies are skipped) and the held-out patches.

Metrics (all fixed in the pre-registration):
  - within_2 / within_4: share of held-out patch nodes in the scoring window whose distance to the nearest fitted
    winding surface (exact point-to-triangle distance, two triangles per mesh quad) is <= 2 / <= 4 voxels. Pooled
    over nodes; the per-patch mean is reported too.
  - winding hops per held-out patch: among nodes within 4 voxels of a winding, the number of grid edges (4-neighbour)
    whose two nodes lie nearest to different windings, the number of distinct windings, and the modal-winding share.
  - duplicate-sheet share: skuzbucket1's geometry_coverage (github.com/skuzbucket1/scroll-prize, MIT,
    experiments/geometry-coverage/geometry_coverage.py at d0b0bdf), re-implemented for one fit's winding folder:
    the area share of mesh nodes that have a node of another winding within 25 voxels, closer than 0.03 mm along
    that winding's normal and less than 25 voxels sideways.
  - satisfied_tracks (from each arm's fit log) and, when given, the fitter's own held-out satisfaction from the eval
    pass (satisfied_fitted.json, 'unverified_patches').
  - B-A paired bootstrap over held-out patches (resample patches with replacement, same draw for both arms).
  - optional winding/plane cuts at fixed z (npz + PNG), drawn over a CT crop when --ct-zarr is given.

usage: score_ab.py --arm A=<meshes dir> --arm B=<meshes dir> --patches <dataset/unverified_patches> --out <json>
       [--prefix hld_] [--z0 9000 --z1 10000 --edge 20] [--log A=<fit log> ...] [--eval-json A=<json> ...]
       [--sections 9250,9500,9750 --ct-zarr <url or path> --ct-level 0] [--boot 10000 --seed 12345]
"""
import argparse
import json
import os
import re
import sys

import numpy as np
from scipy.spatial import cKDTree

VOXEL_UM = 9.362


# --------------------------------------------------------------------------------------------------------------
# input


def load_tifxyz(path):
    import tifffile
    zyx = np.stack([tifffile.imread(os.path.join(path, f'{c}.tif')).astype(np.float64) for c in 'zyx'], axis=-1)
    valid = np.all(zyx != -1, axis=-1) & np.all(np.isfinite(zyx), axis=-1)
    return zyx, valid


_WINDING_RE = re.compile(r'^w(\d+)(_spliced)?(_.*)?$')


def load_windings(meshes_dir, include_spliced=False):
    """[(winding index, folder name, zyx grid, valid mask)] for the raw windings of one fit, by winding index."""
    out = []
    for name in sorted(os.listdir(meshes_dir)):
        m = _WINDING_RE.match(name)
        path = os.path.join(meshes_dir, name)
        if not m or not os.path.isdir(path) or not os.path.exists(os.path.join(path, 'x.tif')):
            continue
        if m.group(2) and not include_spliced:
            continue
        zyx, valid = load_tifxyz(path)
        out.append((int(m.group(1)), name, zyx, valid))
    if not out:
        raise RuntimeError(f'no wNNN tifxyz windings in {meshes_dir}')
    return sorted(out, key=lambda t: t[0])


def load_patches(patch_dir, prefix):
    out = {}
    for name in sorted(os.listdir(patch_dir)):
        if name.startswith(prefix) and os.path.isdir(os.path.join(patch_dir, name)):
            out[name] = load_tifxyz(os.path.join(patch_dir, name))
    if not out:
        raise RuntimeError(f'no {prefix}* patches in {patch_dir}')
    return out


# --------------------------------------------------------------------------------------------------------------
# geometry


def mesh_triangles(windings):
    """Two triangles per valid quad of every winding: vertices (T, 3, 3), winding index (T,), quad centre (T, 3) and
    the largest centre-to-vertex distance of the quad (T,)."""
    tris, wid, centres, radii = [], [], [], []
    for w, _, zyx, valid in windings:
        q = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
        if not q.any():
            continue
        v00, v10, v01, v11 = zyx[:-1, :-1][q], zyx[1:, :-1][q], zyx[:-1, 1:][q], zyx[1:, 1:][q]
        c = (v00 + v10 + v01 + v11) / 4
        r = np.max([np.linalg.norm(v - c, axis=1) for v in (v00, v10, v01, v11)], axis=0)
        tris += [np.stack([v00, v10, v11], 1), np.stack([v00, v11, v01], 1)]
        wid += [np.full(len(c), w)] * 2
        centres += [c, c]
        radii += [r, r]
    return np.concatenate(tris), np.concatenate(wid), np.concatenate(centres), np.concatenate(radii)


def point_triangle_distance(p, a, b, c):
    """Distance from points p (N, 3) to triangles (a, b, c) (N, 3 each); closest-point regions after Ericson,
    Real-Time Collision Detection 5.1.5. Degenerate triangles fall back to the nearest edge/vertex."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - b
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - c
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    out = np.empty_like(p)
    done = np.zeros(len(p), dtype=bool)

    def put(mask, q):
        m = mask & ~done
        out[m] = q[m]
        done[m] = True

    with np.errstate(divide='ignore', invalid='ignore'):
        put((d1 <= 0) & (d2 <= 0), a)
        put((d3 >= 0) & (d4 <= d3), b)
        put((d6 >= 0) & (d5 <= d6), c)
        t = d1 / (d1 - d3)
        put((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + t[:, None] * ab)
        t = d2 / (d2 - d6)
        put((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + t[:, None] * ac)
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), b + t[:, None] * (c - b))
        denom = va + vb + vc
        v, w = vb / denom, vc / denom
        inside = a + v[:, None] * ab + w[:, None] * ac
        put(np.isfinite(inside).all(1), inside)
    if not done.all():   # degenerate leftovers: nearest of the three vertices
        m = ~done
        cand = np.stack([a[m], b[m], c[m]], 1)
        k = np.argmin(np.linalg.norm(cand - p[m][:, None], axis=2), axis=1)
        out[m] = cand[np.arange(m.sum()), k]
    return np.linalg.norm(p - out, axis=1)


def nearest_winding_surface(points, mesh, max_dist=4.0, radius_cap=60.0):
    """Exact distance to the nearest winding surface for points within max_dist of one (else inf), and that winding's
    index (-1 if none). Candidate quads are those whose centre lies within max_dist + their size of the point, so
    every distance <= max_dist is exact."""
    tris, wid, centres, radii = mesh
    keep = radii <= radius_cap
    dropped = int((~keep).sum() // 2)
    tris, wid, centres = tris[keep], wid[keep], centres[keep]
    tree = cKDTree(centres)
    lists = tree.query_ball_point(points, max_dist + float(radii[keep].max()))
    pi = np.repeat(np.arange(len(points)), [len(l) for l in lists])
    ti = np.fromiter((j for l in lists for j in l), dtype=np.int64, count=len(pi))
    dist = np.full(len(points), np.inf)
    win = np.full(len(points), -1)
    if len(pi):
        d = point_triangle_distance(points[pi], tris[ti, 0], tris[ti, 1], tris[ti, 2])
        order = np.lexsort((d, pi))
        pi, ti, d = pi[order], ti[order], d[order]
        first = np.r_[0, np.flatnonzero(np.diff(pi)) + 1]
        best = d[first] <= max_dist
        dist[pi[first][best]] = d[first][best]
        win[pi[first][best]] = wid[ti[first][best]]
    return dist, win, dropped


# --------------------------------------------------------------------------------------------------------------
# metrics


def heldout_metrics(patches, mesh, z0, z1, edge):
    """Per held-out patch: nodes in [z0 + edge, z1 - edge), within-2/4 counts and winding hops."""
    rows = {}
    for name, (zyx, valid) in patches.items():
        inwin = valid & (zyx[..., 0] >= z0 + edge) & (zyx[..., 0] < z1 - edge)
        pts = zyx[inwin]
        dist, win, dropped = nearest_winding_surface(pts, mesh, max_dist=4.0)
        grid_w = np.full(valid.shape, -1)
        grid_w[inwin] = win
        hops = 0
        for a, b in ((grid_w[:, :-1], grid_w[:, 1:]), (grid_w[:-1, :], grid_w[1:, :])):
            both = (a >= 0) & (b >= 0)
            hops += int((both & (a != b)).sum())
        matched = win[win >= 0]
        if len(matched):
            ids, counts = np.unique(matched, return_counts=True)
            modal, modal_share, distinct = int(ids[counts.argmax()]), float(counts.max() / len(matched)), len(ids)
        else:
            modal, modal_share, distinct = None, 0.0, 0
        rows[name] = dict(
            nodes=int(len(pts)), within_2=int((dist <= 2).sum()), within_4=int((dist <= 4).sum()),
            winding_hops=hops, distinct_windings=distinct, modal_winding=modal, modal_share=round(modal_share, 4),
            median_dist_matched=float(np.median(dist[np.isfinite(dist)])) if np.isfinite(dist).any() else None,
            quads_over_radius_cap=dropped)
    return rows


def duplicate_sheet_share(windings, um=VOXEL_UM, same_mm=0.03, lateral=25.0, k=12):
    """skuzbucket1's geometry_coverage on one fit's windings (see the module docstring)."""
    P, N, Wi, area = [], [], [], []
    for i, (w, name, zyx, _) in enumerate(windings):
        Z, Y, X = zyx[..., 0], zyx[..., 1], zyx[..., 2]
        ok = (X > 0) & (Y > 0) & (Z > 0)
        if min(X.shape) < 3:
            area.append(0.0)
            continue
        Xi, Xj = np.gradient(X); Yi, Yj = np.gradient(Y); Zi, Zj = np.gradient(Z)
        nx, ny, nz = Yi * Zj - Zi * Yj, Zi * Xj - Xi * Zj, Xi * Yj - Yi * Xj
        nn = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
        good = ok & (nn > 0) & np.isfinite(nn)
        good[1:-1, 1:-1] &= ok[:-2, 1:-1] & ok[2:, 1:-1] & ok[1:-1, :-2] & ok[1:-1, 2:]
        good[[0, -1], :] = False
        good[:, [0, -1]] = False
        P.append(np.stack([X[good], Y[good], Z[good]], 1))
        N.append(np.stack([nx[good], ny[good], nz[good]], 1) / nn[good][:, None])
        Wi.append(np.full(int(good.sum()), i))
        area.append(float(nn[good].sum()) * (um / 1000.) ** 2)
    P, N, Wi = np.concatenate(P), np.concatenate(N), np.concatenate(Wi)
    same_vx = same_mm * 1000 / um
    dist, idx = cKDTree(P).query(P, k=k, distance_upper_bound=lateral)
    dup_of = np.full(len(P), -1)
    for c in range(1, k):
        j = idx[:, c]
        have = np.isfinite(dist[:, c]) & (dup_of < 0)
        jj = np.where(have, j, 0)
        other = have & (Wi[jj] != Wi)
        d = P[jj] - P
        perp = np.abs((d * N[jj]).sum(1))
        lat = np.sqrt(np.maximum((d * d).sum(1) - perp ** 2, 0))
        hit = other & (perp < same_vx) & (lat < lateral)
        dup_of[hit] = Wi[jj[hit]]
    per_w = []
    for i, (w, name, _, _) in enumerate(windings):
        m = Wi == i
        per_w.append((name, int(m.sum()), area[i], float((dup_of[m] >= 0).mean()) if m.any() else 0.0))
    tot = sum(r[2] for r in per_w)
    shared = sum(r[2] * r[3] for r in per_w) / max(tot, 1e-9)
    return dict(shared_share=shared, area_cm2=tot / 100, windings=len(windings), nodes=int(len(P)),
                same_vx=same_vx, most_shared=sorted(((r[0], round(r[3], 3)) for r in per_w), key=lambda t: -t[1])[:10])


def parse_fit_log(path):
    out = {}
    text = open(path, encoding='utf-8', errors='replace').read()
    for key in ('satisfied_tracks', 'satisfied_track_points', 'unverified_satisfied_patches', 'unverified_satisfied_area'):
        m = re.findall(rf'^{key} = ([\d.]+)/([\d.]+)', text, flags=re.M)
        if m:
            a, b = map(float, m[-1])
            out[key] = dict(n=a, of=b, share=a / b if b else None)
    m = re.findall(r'lasagna normals: resident pool ([\d,]+)/([\d,]+) bricks', text)
    if m:
        out['normals_pool_bricks'] = int(m[-1][0].replace(',', ''))
    m = re.findall(r'loaded (\d+) tracks within z-roi', text)
    if m:
        out['tracks_loaded'] = int(m[-1])
    m = re.findall(r'^ loaded (\d+) unverified patches', text, flags=re.M)
    if m:
        out['unverified_patches_loaded'] = int(m[-1])
    return out


def pooled(rows, names, key):
    n = sum(rows[p]['nodes'] for p in names)
    return sum(rows[p][key] for p in names) / n if n else float('nan')


def paired_bootstrap(rows_a, rows_b, key, boot, seed):
    names = sorted(rows_a)
    na = np.array([rows_a[p]['nodes'] for p in names], dtype=float)
    ka = np.array([rows_a[p][key] for p in names], dtype=float)
    nb = np.array([rows_b[p]['nodes'] for p in names], dtype=float)
    kb = np.array([rows_b[p][key] for p in names], dtype=float)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(names), size=(boot, len(names)))
    with np.errstate(invalid='ignore', divide='ignore'):
        diff = kb[draw].sum(1) / nb[draw].sum(1) - ka[draw].sum(1) / na[draw].sum(1)
    diff = diff[np.isfinite(diff)]
    point = kb.sum() / nb.sum() - ka.sum() / na.sum()
    return dict(point=float(point), ci95=[float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                boot=int(boot), seed=int(seed), patches=len(names))


# --------------------------------------------------------------------------------------------------------------
# cross-sections


def plane_cut(zyx, valid, z):
    """Segments (S, 2, 2) in (y, x) where the triangulated grid crosses the plane z."""
    q = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
    v00, v10, v01, v11 = zyx[:-1, :-1][q], zyx[1:, :-1][q], zyx[:-1, 1:][q], zyx[1:, 1:][q]
    segs = []
    for a, b, c in ((v00, v10, v11), (v00, v11, v01)):
        cross, pts = [], []
        for p, r in ((a, b), (b, c), (c, a)):
            dp, dr = p[:, 0] - z, r[:, 0] - z
            with np.errstate(invalid='ignore', divide='ignore'):
                t = dp / (dp - dr)
                pts.append(p[:, 1:] + t[:, None] * (r[:, 1:] - p[:, 1:]))
            cross.append(dp * dr < 0)
        cross, pts = np.stack(cross, 1), np.stack(pts, 1)          # (T, 3), (T, 3, 2)
        m = cross.sum(1) == 2
        first_two = np.argsort(~cross[m], axis=1, kind='stable')[:, :2]
        rows = np.arange(m.sum())
        segs.append(np.stack([pts[m][rows, first_two[:, 0]], pts[m][rows, first_two[:, 1]]], 1))
    return np.concatenate(segs) if segs else np.empty((0, 2, 2))


def read_ct_crop(url, level, z, y0, y1, x0, x1):
    import zarr
    if url.startswith(('http://', 'https://', 's3://')):
        store = zarr.storage.FsspecStore.from_url(url.rstrip('/') + f'/{level}', read_only=True)
        arr = zarr.open_array(store, mode='r', zarr_format=2)
    else:
        arr = zarr.open_array(os.path.join(url, str(level)), mode='r')
    s = 2 ** level
    return np.asarray(arr[int(z // s), int(y0 // s):int(y1 // s), int(x0 // s):int(x1 // s)])


def write_sections(arms, patches, zs, out_prefix, ct_url=None, ct_level=0, margin=256):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    allp = np.concatenate([zyx[valid] for zyx, valid in patches.values()])
    y0, x0 = allp[:, 1].min() - margin, allp[:, 2].min() - margin
    y1, x1 = allp[:, 1].max() + margin, allp[:, 2].max() + margin
    saved = {}
    for z in zs:
        fig, ax = plt.subplots(figsize=(10, 10 * (y1 - y0) / max(x1 - x0, 1)))
        if ct_url:
            img = read_ct_crop(ct_url, ct_level, z, y0, y1, x0, x1)
            ax.imshow(img, cmap='gray', extent=(x0, x1, y1, y0))
        for (arm, windings), colour in zip(arms.items(), ('tab:orange', 'tab:cyan', 'tab:green')):
            segs = np.concatenate([plane_cut(zyx, v, z) for _, _, zyx, v in windings])
            saved[f'{arm}_z{z}'] = segs
            for s in segs:
                ax.plot(s[:, 1], s[:, 0], color=colour, lw=0.6)
            ax.plot([], [], color=colour, label=f'arm {arm}')
        for name, (zyx, v) in patches.items():
            segs = plane_cut(zyx, v, z)
            saved[f'{name}_z{z}'] = segs
            for s in segs:
                ax.plot(s[:, 1], s[:, 0], color='magenta', lw=1.2)
        ax.plot([], [], color='magenta', label='held-out patches')
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_title(f'z = {z}'); ax.legend(loc='upper right')
        fig.savefig(f'{out_prefix}_z{z}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    np.savez_compressed(f'{out_prefix}_sections.npz', **saved)
    return [f'{out_prefix}_z{z}.png' for z in zs]


# --------------------------------------------------------------------------------------------------------------


def kv(items):
    out = {}
    for item in items or []:
        k, _, v = item.partition('=')
        out[k] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--arm', action='append', required=True, metavar='NAME=MESHES_DIR')
    ap.add_argument('--patches', required=True)
    ap.add_argument('--prefix', default='hld_')
    ap.add_argument('--out', required=True)
    ap.add_argument('--z0', type=int, default=9000)
    ap.add_argument('--z1', type=int, default=10000)
    ap.add_argument('--edge', type=int, default=20, help='score patch nodes in [z0 + edge, z1 - edge)')
    ap.add_argument('--um', type=float, default=VOXEL_UM)
    ap.add_argument('--log', action='append', metavar='NAME=FIT_LOG')
    ap.add_argument('--eval-json', action='append', metavar='NAME=SATISFIED_JSON')
    ap.add_argument('--sections', default='', help='comma-separated z of the fixed cross-sections')
    ap.add_argument('--ct-zarr', default=None)
    ap.add_argument('--ct-level', type=int, default=0)
    ap.add_argument('--boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--go-within2-points', type=float, default=5.0)
    ap.add_argument('--go-dup-points', type=float, default=2.0)
    a = ap.parse_args(argv)

    arm_dirs = kv(a.arm)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    patches = load_patches(a.patches, a.prefix)
    result = dict(prefix=a.prefix, patches=sorted(patches), z_window=[a.z0 + a.edge, a.z1 - a.edge], arms={})
    windings_by_arm = {}
    for arm, d in arm_dirs.items():
        windings = load_windings(d)
        windings_by_arm[arm] = windings
        mesh = mesh_triangles(windings)
        rows = heldout_metrics(patches, mesh, a.z0, a.z1, a.edge)
        names = sorted(rows)
        res = dict(
            meshes=d, windings=len(windings),
            within_2=pooled(rows, names, 'within_2'), within_4=pooled(rows, names, 'within_4'),
            within_2_patch_mean=float(np.mean([rows[p]['within_2'] / rows[p]['nodes'] for p in names if rows[p]['nodes']])),
            within_4_patch_mean=float(np.mean([rows[p]['within_4'] / rows[p]['nodes'] for p in names if rows[p]['nodes']])),
            winding_hops_total=int(sum(r['winding_hops'] for r in rows.values())),
            patches_with_hops=int(sum(r['winding_hops'] > 0 for r in rows.values())),
            duplicate_sheet=duplicate_sheet_share(windings, a.um),
            per_patch=rows)
        result['arms'][arm] = res
        print(f'arm {arm}: {len(windings)} windings; held-out within 2 vox {res["within_2"]:.1%}, within 4 vox '
              f'{res["within_4"]:.1%}; hops {res["winding_hops_total"]} in {res["patches_with_hops"]} patches; '
              f'duplicate-sheet share {res["duplicate_sheet"]["shared_share"]:.1%}', flush=True)
    for arm, path in kv(a.log).items():
        result['arms'].setdefault(arm, {})['fit_log'] = parse_fit_log(path)
    for arm, path in kv(a.eval_json).items():
        entries = json.load(open(path)).get('unverified_patches', [])
        sat = sum(e['satisfied_area'] for e in entries)
        tot = sum(e['total_area'] for e in entries)
        result['arms'].setdefault(arm, {})['fitter_heldout'] = dict(
            satisfied_area_share=sat / tot if tot else None, patches=len(entries),
            per_patch={e['id']: round(e['fraction'], 4) for e in entries})
    if 'A' in result['arms'] and 'B' in result['arms'] and 'per_patch' in result['arms']['A']:
        ra, rb = result['arms']['A']['per_patch'], result['arms']['B']['per_patch']
        cmp_ = {k: paired_bootstrap(ra, rb, k, a.boot, a.seed) for k in ('within_2', 'within_4')}
        dup = (result['arms']['B']['duplicate_sheet']['shared_share']
               - result['arms']['A']['duplicate_sheet']['shared_share'])
        cmp_['duplicate_sheet_share_B_minus_A'] = dup
        w2 = cmp_['within_2']
        go = (w2['point'] * 100 >= a.go_within2_points and w2['ci95'][0] > 0 and dup * 100 <= a.go_dup_points)
        cmp_['GO'] = bool(go)
        cmp_['go_rule'] = (f'B-A within_2 >= +{a.go_within2_points:g} points, its 95% CI above 0, and duplicate-sheet '
                           f'share B-A <= +{a.go_dup_points:g} points')
        result['B_minus_A'] = cmp_
        print(f'B-A within 2 vox {w2["point"] * 100:+.1f} points (95% CI {w2["ci95"][0] * 100:+.1f} to '
              f'{w2["ci95"][1] * 100:+.1f}); duplicate share {dup * 100:+.1f} points -> {"GO" if go else "NO-GO"}')
    if a.sections:
        zs = [int(z) for z in a.sections.split(',')]
        result['sections'] = write_sections(windings_by_arm, patches, zs, os.path.splitext(a.out)[0],
                                            ct_url=a.ct_zarr, ct_level=a.ct_level)
    with open(a.out, 'w') as f:
        json.dump(result, f, indent=1, default=float)
    return result


if __name__ == '__main__':
    main()
