"""Predict, on CPU and before any fit, the 'lasagna normals: resident pool N/M bricks' line the fit will print for its
z-ROI, by running the brick selection of the fitter's own pack_arrays (villa 0fb38c45 pack_resident_pools.py: a brick
is kept when any channel has a non-zero voxel) over the mirrored nx/ny chunks. Guards against the empty-pool trap (a
stale sidecar built for another band loads 1 brick and leaves dense_normals at 0).

By default nothing is written: the fit builds the sidecar itself at start (lasagna_data._ensure_sidecar), which keeps
~0.8 GB off C: until a GPU job needs it. --pack builds it now with pack_arrays (4 IO threads instead of the fitter's 16).

usage: pack_lasagna.py <spiral dir> <dataset dir> --z0 9000 --z1 10000 [--group 2 --scale 4] [--pack]
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('spiral_dir')
    ap.add_argument('dataset')
    ap.add_argument('--z0', type=int, required=True)
    ap.add_argument('--z1', type=int, required=True)
    ap.add_argument('--group', default='2')
    ap.add_argument('--scale', type=int, default=4)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--pack', action='store_true')
    a = ap.parse_args()
    sys.path.insert(0, a.spiral_dir)
    import pack_resident_pools as prp

    nx = os.path.join(a.dataset, 'lasagna_inputs', 'las_008_nx.ome.zarr')
    ny = os.path.join(a.dataset, 'lasagna_inputs', 'las_008_ny.ome.zarr')
    sidecar = prp.sidecar_path(nx, a.group, pair=True)
    dirs = [Path(nx) / a.group, Path(ny) / a.group]
    metas = [prp._read_zarray_meta(d) for d in dirs]
    shape, chunks = tuple(metas[0]['shape']), tuple(metas[0]['chunks'])
    assert all(tuple(m['shape']) == shape and tuple(m['chunks']) == chunks for m in metas)
    seps = [m.get('dimension_separator', '.') for m in metas]
    per_channel = [prp._list_chunk_keys(d, s) for d, s in zip(dirs, seps)]
    keys = sorted(set().union(*per_channel))
    readers = [prp._make_chunk_reader(m['compressor'], int(np.prod(chunks))) for m in metas]

    if a.pack:
        prp.pack_arrays([str(d) for d in dirs], sidecar, label='lasagna normals', io_threads=a.threads)
        meta = json.load(open(os.path.join(sidecar, 'meta.json')))
        coords = np.load(os.path.join(sidecar, 'brick_coords.npy'))
        brick_z, rows = int(meta['brick_shape'][0]), int(meta['rows'])
    else:
        # brick = source chunk (pack_arrays' default); keep a chunk when any channel has a non-zero voxel
        def occupied(key):
            return any(key in ck and r(prp._chunk_path(d, key, s)).any()
                       for d, s, r, ck in zip(dirs, seps, readers, per_channel))
        with ThreadPoolExecutor(a.threads) as pool:
            kept = [k for k, occ in zip(keys, pool.map(occupied, keys, chunksize=64)) if occ]
        coords = np.array([(-1, -1, -1)] + kept, dtype=np.int64)
        brick_z, rows = chunks[0], len(coords)

    z_size = shape[0]
    z_lo = max(0, int(np.floor(a.z0 / a.scale)))
    z_hi = min(z_size, int(np.ceil(a.z1 / a.scale)))
    bz = coords[:, 0].astype(np.int64)
    keep = (bz * brick_z < z_hi) & ((bz + 1) * brick_z > z_lo)      # sparse_cuda_cache.ResidentBrickPool
    keep[0] = True
    rows_present = sorted(set(int(k[0]) for k in keys))
    out = dict(mode='packed' if a.pack else 'predicted (no sidecar written)', sidecar=sidecar,
               chunk_rows_mirrored=[rows_present[0], rows_present[-1]], chunks_listed=len(keys),
               chunks_per_channel=[len(c) for c in per_channel], roi_lasagna_z=[z_lo, z_hi],
               predicted_resident_bricks=int(keep.sum()), predicted_rows=rows,
               predicted_log_line=f'lasagna normals: resident pool {int(keep.sum()):,}/{rows:,} bricks',
               predicted_pool_gib=round(int(keep.sum()) * int(np.prod(chunks)) * 2 / 2 ** 30, 2),
               sidecar_gib_when_built=round(rows * int(np.prod(chunks)) * 2 / 2 ** 30, 2))
    json.dump(out, open(os.path.join(a.dataset, 'lasagna_inputs', 'respool_check.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))
    if out['predicted_resident_bricks'] <= 1000:
        print('POOL_TOO_SMALL')
        sys.exit(1)
    print('POOL_OK')


if __name__ == '__main__':
    main()
