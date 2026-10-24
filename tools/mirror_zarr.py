# Mirror of an open-data zarr v2 array (chunk-row ranges), copied from the First Letters band preparation;
# unchanged apart from these two lines.
"""Mirror part of an open-data zarr v2 array (one resolution group) to a local store: metadata files of the group and
its parent, plus every chunk whose z index falls in [z0, z1) (array-level z, in this group's voxels). Missing chunks
(404, e.g. empty space) are skipped and read back as the fill value. Resumable (existing chunk files are kept).
usage: mirror_zarr.py <bucket path of the zarr root> <group> <local zarr root> --z0 .. --z1 .. [--workers 48]
e.g.   mirror_zarr.py PHerc0813/representations/predictions/lasagna/<run>/PHerc0813_nx.ome.zarr 2 <local> --z0 2825 --z1 3300"""
import argparse, json, os, sys, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

BUCKET = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com'


def get(url, tries=5):
    for t in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            if t == tries - 1:
                raise
        except Exception:
            if t == tries - 1:
                raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root'); ap.add_argument('group'); ap.add_argument('local')
    ap.add_argument('--z0', type=int, required=True); ap.add_argument('--z1', type=int, required=True)
    ap.add_argument('--workers', type=int, default=48)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.local, a.group), exist_ok=True)
    for rel in ('.zgroup', '.zattrs', f'{a.group}/.zarray', f'{a.group}/.zattrs'):
        b = get(f'{BUCKET}/{a.root}/{rel}')
        if b is not None:
            open(os.path.join(a.local, rel), 'wb').write(b)
    m = json.load(open(os.path.join(a.local, a.group, '.zarray')))
    sep = m.get('dimension_separator', '.')
    (Z, Y, X), (cz, cy, cx) = m['shape'], m['chunks']
    keys = [(k, j, i) for k in range(a.z0 // cz, (min(a.z1, Z) - 1) // cz + 1)
            for j in range((Y + cy - 1) // cy) for i in range((X + cx - 1) // cx)]
    done = [0, 0]

    def one(key):
        name = sep.join(map(str, key))
        p = os.path.join(a.local, a.group, *name.split('/'))
        if os.path.exists(p):
            done[0] += 1; return
        b = get(f'{BUCKET}/{a.root}/{a.group}/{name}')
        if b is not None:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p + '.part', 'wb').write(b); os.replace(p + '.part', p); done[1] += len(b)
        done[0] += 1
        if done[0] % 5000 == 0:
            print(f'{done[0]}/{len(keys)} chunks, {done[1] / 1e9:.2f} GB', flush=True)

    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(one, keys))
    print(f'done {a.root}/{a.group} z[{a.z0},{a.z1}) {len(keys)} keys, {done[1] / 1e9:.2f} GB new', flush=True)


if __name__ == '__main__':
    main()
