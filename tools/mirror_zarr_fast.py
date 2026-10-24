"""Mirror the z rows [z0, z1) of one resolution group of an open-data zarr v2 array, like tools/mirror_zarr.py
(same local layout, resumable, metadata of the group and its parent), but list each chunk row with S3 ListObjectsV2
first so absent (empty-space) chunks cost no request, and reuse one keep-alive HTTP session per thread.
usage: mirror_zarr_fast.py <bucket path of the zarr root> <group> <local zarr root> --z0 .. --z1 .. [--workers 4]
"""
import argparse
import json
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import requests

BUCKET = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com'
_local = threading.local()


def session():
    if not hasattr(_local, 's'):
        _local.s = requests.Session()
    return _local.s


def get(url, tries=6):
    for t in range(tries):
        try:
            r = session().get(url, timeout=120)
            if r.status_code in (403, 404):
                return None
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if t == tries - 1:
                raise
            time.sleep(2 * (t + 1))


def list_keys(prefix):
    keys, token = [], None
    ns = '{http://s3.amazonaws.com/doc/2006-03-01/}'
    while True:
        params = {'list-type': '2', 'prefix': prefix, 'max-keys': '1000'}
        if token:
            params['continuation-token'] = token
        r = session().get(BUCKET + '/', params=params, timeout=120)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        keys += [(c.find(ns + 'Key').text, int(c.find(ns + 'Size').text)) for c in root.iter(ns + 'Contents')]
        if root.find(ns + 'IsTruncated').text != 'true':
            return keys
        token = root.find(ns + 'NextContinuationToken').text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root'); ap.add_argument('group'); ap.add_argument('local')
    ap.add_argument('--z0', type=int, required=True); ap.add_argument('--z1', type=int, required=True)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(os.path.join(a.local, a.group), exist_ok=True)
    for rel in ('.zgroup', '.zattrs', f'{a.group}/.zarray', f'{a.group}/.zattrs'):
        b = get(f'{BUCKET}/{a.root}/{rel}')
        if b is not None:
            open(os.path.join(a.local, rel), 'wb').write(b)
    m = json.load(open(os.path.join(a.local, a.group, '.zarray')))
    assert m.get('dimension_separator', '.') == '/', 'row listing assumes "/"-separated chunk keys'
    Z, cz = m['shape'][0], m['chunks'][0]
    rows = range(a.z0 // cz, (min(a.z1, Z) - 1) // cz + 1)
    with ThreadPoolExecutor(a.workers) as pool:
        listed = list(pool.map(lambda k: list_keys(f'{a.root}/{a.group}/{k}/'), rows))
    todo, n_listed, listed_bytes = [], 0, 0
    pat = re.compile(r'/(\d+)/(\d+)/(\d+)$')
    for keys in listed:
        for key, size in keys:
            if not pat.search(key):
                continue
            n_listed += 1; listed_bytes += size
            rel = key[len(a.root) + 1:]
            p = os.path.join(a.local, *rel.split('/'))
            if not (os.path.exists(p) and os.path.getsize(p) == size):
                todo.append((key, p, size))
    print(f'{a.root}/{a.group}: rows {rows.start}..{rows.stop - 1}, {n_listed} chunks listed '
          f'({listed_bytes / 1e9:.2f} GB), {len(todo)} to fetch', flush=True)
    done = [0, 0]

    def one(item):
        key, p, size = item
        b = get(f'{BUCKET}/{key}')
        if b is None or len(b) != size:
            raise IOError(f'{key}: got {None if b is None else len(b)} bytes, listed {size}')
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p + '.part', 'wb') as f:
            f.write(b)
        os.replace(p + '.part', p)
        done[0] += 1; done[1] += len(b)
        if done[0] % 2000 == 0:
            print(f'{done[0]}/{len(todo)} chunks, {done[1] / 1e9:.2f} GB ({time.time() - t0:.0f}s)', flush=True)

    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(one, todo))
    print(f'done {a.root}/{a.group} z[{a.z0},{a.z1}) {n_listed} chunks present, {done[1] / 1e9:.2f} GB new '
          f'({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
