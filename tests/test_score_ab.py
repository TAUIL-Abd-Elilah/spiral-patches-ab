"""Synthetic checks of score_ab.py: concentric cylinder 'windings', patches on known windings.
Run: python tests/test_score_ab.py   (or pytest)"""
import json
import os
import sys
import tempfile

import numpy as np
import tifffile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import score_ab as S  # noqa: E402

CY, CX = 500.0, 500.0


def write_tifxyz(path, zyx):
    os.makedirs(path, exist_ok=True)
    for i, c in enumerate('zyx'):
        tifffile.imwrite(os.path.join(path, f'{c}.tif'), zyx[..., i].astype(np.float32))
    json.dump({'format': 'tifxyz', 'scale': [0.05, 0.05], 'uuid': os.path.basename(path)},
              open(os.path.join(path, 'meta.json'), 'w'))


def cylinder(r, z0=9000, z1=10000, step=20.0, th0=0.0, th1=2 * np.pi, dz=0.0):
    zs = np.arange(z0, z1 + 1e-6, step) + dz
    n = max(2, int(np.ceil((th1 - th0) * r / step)) + 1)
    th = np.linspace(th0, th1, n)
    Z, T = np.meshgrid(zs, th, indexing='ij')
    return np.stack([Z, CY + r * np.sin(T), CX + r * np.cos(T)], -1)


def make_fit(root, tag, radii, spliced_offset=None):
    d = os.path.join(root, f'fitted_{tag}')
    for k, r in enumerate(radii):
        write_tifxyz(os.path.join(d, f'w{10 + k:03d}_{tag}'), cylinder(r))
        if spliced_offset is not None:   # must be ignored by the scorer
            write_tifxyz(os.path.join(d, f'w{10 + k:03d}_spliced_{tag}'), cylinder(r + spliced_offset))
    return d


def make_patches(root):
    d = os.path.join(root, 'unverified_patches')
    # on winding 2 (r = 240), z 9100-9800, theta 0..0.5
    write_tifxyz(os.path.join(d, 'hld_on240.tifxyz'), cylinder(240, 9100, 9800, th0=0.0, th1=0.5))
    # left half on r = 240, right half on r = 260: one hop per row
    p = cylinder(240, 9100, 9800, th0=1.0, th1=1.6)
    q = cylinder(260, 9100, 9800, th0=1.0, th1=1.6)
    half = p.shape[1] // 2
    p[:, half:] = q[:, half:p.shape[1]]
    write_tifxyz(os.path.join(d, 'hld_hop.tifxyz'), p)
    # a training patch far from the right radius: must not be scored with --prefix hld_
    write_tifxyz(os.path.join(d, 'trn_far.tifxyz'), cylinder(250, 9100, 9800, th0=3.0, th1=3.5))
    return d


def test_point_triangle_distance_matches_brute_force():
    rng = np.random.default_rng(0)
    n = 300
    a, b, c = (rng.normal(size=(n, 3)) * 10 for _ in range(3))
    p = rng.normal(size=(n, 3)) * 15
    d = S.point_triangle_distance(p, a, b, c)
    u, v = np.meshgrid(np.linspace(0, 1, 301), np.linspace(0, 1, 301))
    keep = u + v <= 1
    u, v = u[keep], v[keep]
    for i in range(0, n, 10):
        pts = a[i] + u[:, None] * (b[i] - a[i]) + v[:, None] * (c[i] - a[i])
        brute = np.linalg.norm(pts - p[i], axis=1).min()
        assert d[i] <= brute + 1e-9 and brute - d[i] < 0.2, (i, d[i], brute)


def test_heldout_distances_and_hops():
    with tempfile.TemporaryDirectory() as tmp:
        radii = [200 + 20 * k for k in range(6)]
        fa = make_fit(tmp, 'A', radii, spliced_offset=7.0)
        fb = make_fit(tmp, 'B', [r + 3 for r in radii])
        patches = S.load_patches(make_patches(tmp), 'hld_')
        assert sorted(patches) == ['hld_hop.tifxyz', 'hld_on240.tifxyz']
        wa = S.load_windings(fa)
        assert len(wa) == 6 and all('_spliced' not in w[1] for w in wa)
        ra = S.heldout_metrics(patches, S.mesh_triangles(wa), 9000, 10000, 20)
        rb = S.heldout_metrics(patches, S.mesh_triangles(S.load_windings(fb)), 9000, 10000, 20)
        on = ra['hld_on240.tifxyz']
        assert on['within_2'] == on['nodes'] > 0 and on['winding_hops'] == 0 and on['distinct_windings'] == 1
        assert on['modal_winding'] == 12
        onb = rb['hld_on240.tifxyz']
        assert onb['within_2'] == 0 and onb['within_4'] == onb['nodes']    # 3 vox off every node
        hop = ra['hld_hop.tifxyz']
        rows = 36                                                           # z 9100..9800 step 20
        assert hop['distinct_windings'] == 2 and hop['winding_hops'] == rows, hop
        assert hop['within_2'] == hop['nodes']


def test_edge_window_and_bootstrap():
    rows_a = {f'p{i}': dict(nodes=100, within_2=50, within_4=80) for i in range(7)}
    rows_b = {f'p{i}': dict(nodes=100, within_2=60 + i, within_4=85) for i in range(7)}
    r = S.paired_bootstrap(rows_a, rows_b, 'within_2', 2000, 1)
    assert abs(r['point'] - 0.13) < 1e-12
    assert r['ci95'][0] <= r['point'] <= r['ci95'][1] and r['ci95'][0] > 0.09 and r['ci95'][1] < 0.17
    assert S.paired_bootstrap(rows_a, rows_b, 'within_2', 2000, 1) == r   # fixed seed, same answer


def test_duplicate_sheet_share():
    with tempfile.TemporaryDirectory() as tmp:
        radii = [200 + 20 * k for k in range(6)]
        clean = S.duplicate_sheet_share(S.load_windings(make_fit(tmp, 'A', radii)))
        assert clean['shared_share'] == 0.0
        # winding 13 copied 1 vox outwards as an extra winding: both copies count as shared
        dup_dir = make_fit(tmp, 'C', radii)
        write_tifxyz(os.path.join(dup_dir, 'w016_C'), cylinder(radii[3] + 1.0))
        dup = S.duplicate_sheet_share(S.load_windings(dup_dir))
        area = {r: r for r in radii + [radii[3] + 1.0]}                      # area ~ radius for equal heights
        expect = (area[radii[3]] + area[radii[3] + 1.0]) / sum(area.values())
        assert abs(dup['shared_share'] - expect) < 0.02, (dup['shared_share'], expect)
        # 5 vox apart is a different sheet at the 0.03 mm (3.2 vox) threshold
        far_dir = make_fit(tmp, 'D', radii)
        write_tifxyz(os.path.join(far_dir, 'w016_D'), cylinder(radii[3] + 5.0))
        assert S.duplicate_sheet_share(S.load_windings(far_dir))['shared_share'] < 0.01


def test_plane_cut_on_cylinder():
    zyx = cylinder(240)
    valid = np.ones(zyx.shape[:2], bool)
    segs = S.plane_cut(zyx, valid, 9510.0)
    r = np.hypot(segs[..., 0] - CY, segs[..., 1] - CX)
    assert len(segs) > 50 and np.all(np.abs(r - 240) < 0.5)


def test_end_to_end_cli():
    with tempfile.TemporaryDirectory() as tmp:
        radii = [200 + 20 * k for k in range(6)]
        fa = make_fit(tmp, 'A', [r + 3 for r in radii])      # A off by 3 vox
        fb = make_fit(tmp, 'B', radii)                       # B on the sheets
        pdir = make_patches(tmp)
        log = os.path.join(tmp, 'fit.log')
        open(log, 'w').write('loaded 1270348 tracks within z-roi [9000, 10000)\n'
                             'lasagna normals: resident pool 12,345/14,000 bricks of (32, 32, 32) x2\n'
                             'satisfied_tracks = 230590/1269517 (18.2%)\n')
        ej = os.path.join(tmp, 'satisfied_fitted.json')
        json.dump({'unverified_patches': [{'id': 'hld_on240.tifxyz', 'satisfied_area': 3.0, 'total_area': 4.0,
                                           'fraction': 0.75}]}, open(ej, 'w'))
        out = os.path.join(tmp, 'score.json')
        res = S.main(['--arm', f'A={fa}', '--arm', f'B={fb}', '--patches', pdir, '--out', out, '--boot', '500',
                      '--log', f'A={log}', '--eval-json', f'B={ej}', '--sections', '9500'])
        cmp_ = res['B_minus_A']
        assert abs(cmp_['within_2']['point'] - 1.0) < 1e-9 and cmp_['GO']
        assert res['arms']['A']['fit_log']['tracks_loaded'] == 1270348
        assert res['arms']['A']['fit_log']['normals_pool_bricks'] == 12345
        assert abs(res['arms']['B']['fitter_heldout']['satisfied_area_share'] - 0.75) < 1e-12
        assert os.path.exists(os.path.join(tmp, 'score_z9500.png'))
        assert json.load(open(out))['B_minus_A']['GO'] is True


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
