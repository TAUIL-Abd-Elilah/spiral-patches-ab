"""Load a band track DBM with the fitter's own loader (villa 0fb38c45 tracks.load_tracks_from_dbm, CPU only, no fit)
and check it against the published packed store: track count, source ids, families and points.

usage: verify_band_tracks.py <spiral dir of the villa worktree> <band dbm> <index cache dir> --z0 9000 --z1 10000
       [--expect 1270348]
"""
import argparse
import json
import os
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('spiral_dir')
    ap.add_argument('dbm')
    ap.add_argument('index_cache', help='z_bounds.i32, offsets.i64, source_ids.u64, family_codes.i8 of the store')
    ap.add_argument('--z0', type=int, required=True)
    ap.add_argument('--z1', type=int, required=True)
    ap.add_argument('--expect', type=int, default=None)
    a = ap.parse_args()
    sys.path.insert(0, a.spiral_dir)
    os.environ.setdefault('VC_DISABLE_NATIVE_TRACK_CROSSINGS', '1')
    import tracks as T

    t0 = time.time()
    loaded, families, source_ids = T.load_tracks_from_dbm(
        a.dbm, a.z0, a.z1, return_families=True, return_source_ids=True, show_progress=False)
    print(f'loader: {len(loaded):,} tracks within z-roi [{a.z0}, {a.z1}) ({time.time() - t0:.0f}s, '
          f'{"packed store" if hasattr(loaded, "coordinates") else "DBM path"})', flush=True)
    n_points = sum(len(t) for t in loaded)
    zmin = np.array([t[:, 0].min() for t in loaded])
    zmax = np.array([t[:, 0].max() for t in loaded])
    del loaded

    zb = np.fromfile(os.path.join(a.index_cache, 'z_bounds.i32'), dtype='<i4').reshape(-1, 2)
    sel = np.flatnonzero((zb[:, 0] >= a.z0) & (zb[:, 1] < a.z1))
    off = np.fromfile(os.path.join(a.index_cache, 'offsets.i64'), dtype='<i8')
    sid = np.fromfile(os.path.join(a.index_cache, 'source_ids.u64'), dtype='<u8')[sel]
    fam = np.fromfile(os.path.join(a.index_cache, 'family_codes.i8'), dtype='<i1')[sel]
    pts = int((off[sel + 1] - off[sel]).sum())

    # the loader walks keys in dbm order; match tracks by source id
    order = np.argsort(source_ids, kind='stable')
    fam_loaded = np.array([0 if f == 'horizontal' else (1 if f == 'vertical' else -1) for f in families])[order]
    checks = {
        'count': len(source_ids),
        'published_count': len(sel),
        'count_equal': len(source_ids) == len(sel),
        'source_ids_equal': bool(np.array_equal(np.sort(source_ids), sid)),
        'families_equal': bool(np.array_equal(fam_loaded, fam)) if len(fam_loaded) == len(fam) else False,
        'points': n_points,
        'points_equal': n_points == pts,
        'z_bounds_equal': (bool(np.array_equal(zmin[order], zb[sel, 0]) and np.array_equal(zmax[order], zb[sel, 1]))
                           if len(order) == len(sel) else False),
        'loader_order_is_source_id_order': bool(np.all(np.diff(source_ids.astype(np.int64)) > 0)),
    }
    if a.expect is not None:
        checks['expected'] = a.expect
        checks['expected_equal'] = len(source_ids) == a.expect
    print(json.dumps(checks, indent=2), flush=True)
    ok = all(v for k, v in checks.items() if k.endswith('_equal'))
    print('VERIFY_OK' if ok else 'VERIFY_FAILED', flush=True)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
