"""Cut a z band out of a published Spiral track store without downloading the whole DBM.

Reads the per-track index arrays of the published packed store (<name>.dbm.vctracks: z_bounds.i32, offsets.i64,
source_ids.u64, family_codes.i8), keeps the tracks lying wholly inside [z0, z1) (the rule load_tracks_from_dbm
applies: zmin >= z0 and zmax < z1), range-reads only their coordinates, and writes a small DBM in the format
extract_surface_tracks.py writes and tracks.load_tracks_from_dbm reads (dbm.sqlite3; one key per extraction chunk,
value = pickled list of (N, 3) int32 zyx arrays).

The band DBM keeps the published key names and each track's position in its key's list, so a track's source id
(key ordinal << 32 | list index) is the same as in the full store. Tracks outside the band are replaced by one shared
empty (0, 3) int32 array (pickle memoises it, so each costs a couple of bytes); every reader skips len() == 0 entries.
Key names are rebuilt from the extraction parameters in <name>.extract.json and checked against the family codes and
the z bounds of the tracks.

usage: cut_band_tracks.py <tracks base url> <store name> <out dbm> --z0 9000 --z1 10000 [--cache DIR] [--workers 4]
"""
import argparse
import json
import os
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import dbm
import dbm.sqlite3  # noqa: F401  (the format the published stores use)
import numpy as np
import requests

_local = threading.local()


def session():
    if not hasattr(_local, 's'):
        _local.s = requests.Session()
    return _local.s


def get(url, byte_range=None, tries=6):
    headers = {} if byte_range is None else {'Range': f'bytes={byte_range[0]}-{byte_range[1] - 1}'}
    for t in range(tries):
        try:
            r = session().get(url, headers=headers, timeout=120)
            if byte_range is None and r.status_code == 200:
                return r.content
            if byte_range is not None and r.status_code == 206 and len(r.content) == byte_range[1] - byte_range[0]:
                return r.content
            raise IOError(f'HTTP {r.status_code}, {len(r.content)} bytes for {url} {byte_range}')
        except (IOError, requests.RequestException) as e:
            if t == tries - 1:
                raise
            print(f'retry {t + 1}: {e}', flush=True)
            time.sleep(2 * (t + 1))


def cached_array(url, cache, name, dtype):
    path = os.path.join(cache, name) if cache else None
    if path and os.path.exists(path):
        return np.fromfile(path, dtype=dtype)
    data = get(url)
    if path:
        with open(path + '.part', 'wb') as f:
            f.write(data)
        os.replace(path + '.part', path)
    return np.frombuffer(data, dtype=dtype).copy()


def extraction_keys(extract, n_h):
    """Key names in extraction (insertion) order: h, then vy, then vx (extract_surface_tracks.main)."""
    p = extract['parameters']
    (y0, x0), (y1, x1) = extract['yx_bounds']
    h = [f'h:{z}' for z in range(p['z_min'], p['z_max'], p['z_chunk_stride_h'])][:n_h]
    vy = [f'vy:{w}' for w in range(y0, y1, p['yx_stride_v'])]
    vx = [f'vx:{w}' for w in range(x0, x1, p['yx_stride_v'])]
    return h + vy + vx


def family_of(key):
    prefix = key.split(':', 1)[0]
    return 0 if prefix == 'h' else (1 if prefix in ('vx', 'vy') else -1)


def match_key_names(extract, ordinals, families, z_bounds, header_tracks):
    """Find the horizontal-key count whose sorted key list agrees with every track's family code, and each h:Z
    key's tracks with z in [Z, Z + z_chunk_depth_h)."""
    p = extract['parameters']
    max_h = len(range(p['z_min'], p['z_max'], p['z_chunk_stride_h']))
    first = np.r_[0, np.flatnonzero(np.diff(ordinals)) + 1]        # first track of each non-empty key
    key_ord = ordinals[first]
    key_fam = families[first]
    # a key's tracks share one family; check it once
    assert np.array_equal(np.repeat(key_fam, np.diff(np.r_[first, len(ordinals)])), families), 'mixed family in a key'
    zmin_k = np.minimum.reduceat(z_bounds[:, 0], first)
    zmax_k = np.maximum.reduceat(z_bounds[:, 1], first)
    good = []
    for n_h in range(max_h, 0, -1):
        keys = sorted(extraction_keys(extract, n_h))
        if len(keys) <= key_ord.max():
            break
        fam = np.array([family_of(keys[o]) for o in key_ord])
        if not np.array_equal(fam, key_fam):
            continue
        hz = np.array([int(keys[o].split(':')[1]) if fam_o == 0 else 0 for o, fam_o in zip(key_ord, fam)])
        h = fam == 0
        if np.all(zmin_k[h] >= hz[h]) and np.all(zmax_k[h] < hz[h] + p['z_chunk_depth_h']):
            good.append(n_h)
    if len(good) != 1:
        raise RuntimeError(f'could not pin the key list: candidates for the h-key count {good}')
    return good[0]


def merged_ranges(starts, ends, gap):
    """Merge sorted [start, end) byte ranges whose gap is <= gap bytes."""
    out = []
    s0, e0 = int(starts[0]), int(ends[0])
    for s, e in zip(starts[1:], ends[1:]):
        if s - e0 <= gap:
            e0 = int(e)
        else:
            out.append((s0, e0))
            s0, e0 = int(s), int(e)
    out.append((s0, e0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base_url')
    ap.add_argument('store_name', help='e.g. PHerc0813_20250821151723_surface_m7_L0_th0.2.dbm')
    ap.add_argument('out_dbm')
    ap.add_argument('--z0', type=int, required=True)
    ap.add_argument('--z1', type=int, required=True)
    ap.add_argument('--cache', default='', help='keep the downloaded index arrays here (resumable)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--merge-gap', type=int, default=16 * 1024)
    ap.add_argument('--keys-per-batch', type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    store = f'{a.base_url}/{a.store_name}.vctracks'
    if a.cache:
        os.makedirs(a.cache, exist_ok=True)

    header = get(f'{store}/header.bin')
    magic, version, header_size, n_tracks, n_points = np.frombuffer(header[:32], dtype=np.dtype(
        [('m', 'S8'), ('v', '<u4'), ('h', '<u4'), ('t', '<u8'), ('p', '<u8')]))[0]
    assert magic == b'VCTRK01' and version == 1, (magic, version)
    meta = json.loads(get(f'{store}/metadata.json'))
    extract = json.loads(get(f'{a.base_url}/{a.store_name[:-4]}.extract.json'))
    print(f'store: {n_tracks:,} tracks, {n_points:,} points; metadata {meta}', flush=True)

    z_bounds = cached_array(f'{store}/z_bounds.i32', a.cache, 'z_bounds.i32', '<i4').reshape(-1, 2)
    offsets = cached_array(f'{store}/offsets.i64', a.cache, 'offsets.i64', '<i8')
    source_ids = cached_array(f'{store}/source_ids.u64', a.cache, 'source_ids.u64', '<u8')
    families = cached_array(f'{store}/family_codes.i8', a.cache, 'family_codes.i8', '<i1')
    assert len(z_bounds) == len(source_ids) == len(families) == n_tracks and len(offsets) == n_tracks + 1
    assert int(offsets[-1]) == n_points
    print(f'index arrays read ({time.time() - t0:.0f}s)', flush=True)

    ordinals = (source_ids >> np.uint64(32)).astype(np.int64)
    indices = (source_ids & np.uint64(0xFFFFFFFF)).astype(np.int64)
    assert np.all(np.diff(ordinals) >= 0), 'packed store is not in key order'
    n_h = match_key_names(extract, ordinals, families, z_bounds, n_tracks)
    keys_insert = extraction_keys(extract, n_h)
    keys_sorted = sorted(keys_insert)
    print(f'key list: {n_h} h + {len(keys_insert) - n_h} v keys = {len(keys_insert)}; '
          f'{len(np.unique(ordinals)):,} keys hold tracks', flush=True)

    sel = np.flatnonzero((z_bounds[:, 0] >= a.z0) & (z_bounds[:, 1] < a.z1))
    n_sel_points = int((offsets[sel + 1] - offsets[sel]).sum())
    print(f'band [{a.z0}, {a.z1}): {len(sel):,} tracks, {n_sel_points:,} points '
          f'({n_sel_points * 12 / 2 ** 30:.2f} GiB of coordinates)', flush=True)

    # selected tracks grouped by key ordinal (contiguous: the store is in key order)
    sel_ord = ordinals[sel]
    bounds = np.r_[0, np.flatnonzero(np.diff(sel_ord)) + 1, len(sel)]
    groups = {int(sel_ord[b]): sel[b:e] for b, e in zip(bounds[:-1], bounds[1:])}

    coords_url = f'{store}/coordinates.i32'
    empty = np.empty((0, 3), dtype=np.int32)       # one shared placeholder for out-of-band list slots
    stats = dict(requests=0, bytes=0, points=0, tracks=0)
    tmp = a.out_dbm + '.part'
    for suffix in ('', '-wal', '-shm'):
        if os.path.exists(tmp + suffix):
            os.remove(tmp + suffix)

    def fetch_key(ordinal):
        tr = groups[ordinal]
        p_start, p_end = offsets[tr], offsets[tr + 1]
        ranges = merged_ranges(p_start * 12, p_end * 12, a.merge_gap)
        blobs = [(s, get(coords_url, (s, e))) for s, e in ranges]
        tracks = []
        bi = 0
        for ps, pe in zip(p_start, p_end):
            bs, be = int(ps) * 12, int(pe) * 12
            while not (blobs[bi][0] <= bs and be <= blobs[bi][0] + len(blobs[bi][1])):
                bi += 1
            b0 = blobs[bi][0]
            arr = np.frombuffer(blobs[bi][1][bs - b0:be - b0], dtype='<i4').reshape(-1, 3).astype(np.int32)
            tracks.append(arr)
        nbytes = sum(len(b) for _, b in blobs)
        return ordinal, tracks, len(ranges), nbytes

    ordered = sorted(groups)
    with ThreadPoolExecutor(a.workers) as pool, dbm.open(tmp, 'n') as db:
        # dbm.sqlite3 iterates keys through the unique key index, i.e. in sorted key order like the published
        # store, so the write order does not matter
        done = 0
        for start in range(0, len(ordered), a.keys_per_batch):
            batch = ordered[start:start + a.keys_per_batch]
            for ordinal, tracks, n_req, nbytes in pool.map(fetch_key, batch):
                tr = groups[ordinal]
                zb = z_bounds[tr]
                got_min = np.array([t[:, 0].min() for t in tracks])
                got_max = np.array([t[:, 0].max() for t in tracks])
                if not (np.array_equal(got_min, zb[:, 0]) and np.array_equal(got_max, zb[:, 1])):
                    raise RuntimeError(f'key ordinal {ordinal}: fetched coordinates disagree with z_bounds')
                slots = [empty] * (int(indices[tr].max()) + 1)
                for i, t in zip(indices[tr], tracks):
                    slots[int(i)] = t
                db[keys_sorted[ordinal]] = pickle.dumps(slots, protocol=4)
                stats['requests'] += n_req
                stats['bytes'] += nbytes
                stats['points'] += sum(len(t) for t in tracks)
                stats['tracks'] += len(tracks)
            done += len(batch)
            if done % 80 < a.keys_per_batch or done == len(ordered):
                print(f'{done}/{len(ordered)} keys, {stats["tracks"]:,} tracks, {stats["requests"]:,} requests, '
                      f'{stats["bytes"] / 2 ** 30:.2f} GiB ({time.time() - t0:.0f}s)', flush=True)
        empty_value = pickle.dumps([], protocol=4)
        for ordinal, key in enumerate(keys_sorted):
            if ordinal not in groups:
                db[key] = empty_value
    assert stats['tracks'] == len(sel) and stats['points'] == n_sel_points
    for suffix in ('-wal', '-shm'):
        if os.path.exists(tmp + suffix):
            raise RuntimeError(f'{tmp + suffix} left behind; the DBM was not closed cleanly')
    os.replace(tmp, a.out_dbm)
    summary = {
        'source_store': f'{a.base_url}/{a.store_name}',
        'source_db_signature': meta.get('source_db_signature'),
        'z_range': [a.z0, a.z1],
        'rule': 'zmin >= z0 and zmax < z1 (tracks.load_tracks_from_dbm)',
        'tracks': int(len(sel)), 'points': n_sel_points,
        'keys': len(keys_insert), 'keys_with_band_tracks': len(groups), 'h_keys': n_h,
        'http_range_requests': stats['requests'], 'bytes_fetched': stats['bytes'],
        'merge_gap_bytes': a.merge_gap,
        'source_ids_sha1': __import__('hashlib').sha1(np.ascontiguousarray(source_ids[sel]).tobytes()).hexdigest(),
        'dbm_bytes': os.path.getsize(a.out_dbm),
        'seconds': round(time.time() - t0),
    }
    with open(a.out_dbm + '.band.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
