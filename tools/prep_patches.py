"""Check the Ahggggg/first-letters-clean-sheets PHerc0813 patches against the fit volume and split them into training
and held-out sets for the patches A/B (PREREG_patches_ab.md).

Split rule (fixed before any fit of this band was run; it uses patch geometry only):
  - drop patches whose index.csv 'flags' column is non-empty;
  - drop patches with no valid node in the band [z0, z1);
  - held-out ('hld_') = patches whose in-band nodes have mean z >= --split-z (9600 for z [9000, 10000));
    training ('trn_') = the rest.
The held-out group is one contiguous z group at the top of the band; the script also reports how far it lies from
the training patches.

Each kept patch is copied unchanged (x/y/z.tif, meta.json) to <dataset>/unverified_patches/<trn_|hld_><patch>.tifxyz;
the folder rename is the only change. The data licence (CC BY-NC 4.0) and attribution go beside them as _*.md files,
which the fitter's patch_uuid_filter_regex ('^trn_' / '^hld_') never loads.

usage: prep_patches.py <repo copy> <dataset dir> --z0 9000 --z1 10000 --split-z 9600
"""
import argparse
import csv
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

VOLUME_ID = '20250821151723'
VOLUME_SHAPE_ZYX = [16993, 7947, 7947]
VOXEL_UM = 9.362


def load_zyx(path):
    return np.stack([np.array(Image.open(f'{path}/{c}.tif')) for c in 'zyx'], axis=-1).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('repo', help='local copy of first-letters-clean-sheets (data/PHerc0813, index.csv, licence files)')
    ap.add_argument('dataset')
    ap.add_argument('--z0', type=int, required=True)
    ap.add_argument('--z1', type=int, required=True)
    ap.add_argument('--split-z', type=float, required=True)
    a = ap.parse_args()

    with open(os.path.join(a.repo, 'index.csv'), newline='') as f:
        index = {r['patch']: r for r in csv.DictReader(f) if r['scroll'] == 'PHerc0813'}
    commit = open(os.path.join(a.repo, 'SOURCE_COMMIT.txt')).readline().strip()
    out_dir = os.path.join(a.dataset, 'unverified_patches')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(a.dataset, 'verified_patches'), exist_ok=True)   # required (empty) while patches are on

    rows, nodes_in_band = [], {}
    for name in sorted(index):
        r = index[name]
        src = os.path.join(a.repo, r['tifxyz'])
        meta = json.load(open(os.path.join(src, 'meta.json')))
        zyx = load_zyx(src)
        valid = np.all(zyx != -1, axis=-1)
        v = zyx[valid]
        # level-0 frame of the fit volume
        checks = {
            'volume_id': meta.get('volume_id') == VOLUME_ID == r['volume_id'],
            'scroll': meta.get('scroll') == 'PHerc0813',
            'voxel_um': abs(float(meta.get('voxel_size_um', 0)) - VOXEL_UM) < 1e-6,
            'volume_shape_L0': meta.get('volume_shape_zyx') == VOLUME_SHAPE_ZYX,
            'scale_2_positive': len(meta.get('scale', [])) == 2 and all(s > 0 for s in meta['scale']),
            'inside_volume': bool(np.all(v >= 0) and np.all(v < np.array(VOLUME_SHAPE_ZYX))),
            'no_nan': bool(np.isfinite(v).all()),
        }
        spacing = np.linalg.norm(np.diff(zyx, axis=1), axis=-1)[valid[:, 1:] & valid[:, :-1]]
        qv = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
        qz = (zyx[:-1, :-1, 0] + zyx[1:, :-1, 0] + zyx[:-1, 1:, 0] + zyx[1:, 1:, 0]) / 4
        q_in = qv & (qz >= a.z0) & (qz < a.z1)
        in_band = v[(v[:, 0] >= a.z0) & (v[:, 0] < a.z1)]
        nodes_in_band[name] = in_band
        step = float(np.median(spacing))
        rows.append(dict(
            patch=name, flags=r['flags'], checks_ok=all(checks.values()), checks=checks,
            grid=list(zyx.shape[:2]), nodes=int(valid.sum()), nodes_in_band=int(len(in_band)),
            quads_in_band=int(q_in.sum()), area_in_band_cm2=round(float(q_in.sum()) * step ** 2 * (VOXEL_UM * 1e-4) ** 2, 3),
            area_cm2_index=float(r['area_cm2']), node_spacing_vox=round(step, 2),
            z_range=[float(v[:, 0].min()), float(v[:, 0].max())],
            z_mean_in_band=round(float(in_band[:, 0].mean()), 1) if len(in_band) else None,
            xsec_median_abs_drift_vox=float(r['xsec_median_abs_drift_vox']),
            xsec_share_sections_drift_ge_20vox=float(r['xsec_share_sections_drift_ge_20vox']),
        ))
    bad = [r['patch'] for r in rows if not r['checks_ok']]
    if bad:
        print('frame check failed:', bad, [r['checks'] for r in rows if not r['checks_ok']])
        sys.exit(1)

    for r in rows:
        if r['flags']:
            r['group'], r['reason'] = 'excluded', f'index.csv flags: {r["flags"]}'
        elif r['nodes_in_band'] == 0:
            r['group'], r['reason'] = 'excluded', 'no node in the band'
        elif r['z_mean_in_band'] >= a.split_z:
            r['group'], r['reason'] = 'hld', f'in-band mean z >= {a.split_z:g}'
        else:
            r['group'], r['reason'] = 'trn', f'in-band mean z < {a.split_z:g}'

    trn = [r['patch'] for r in rows if r['group'] == 'trn']
    hld = [r['patch'] for r in rows if r['group'] == 'hld']
    tree = cKDTree(np.concatenate([nodes_in_band[p] for p in trn]))
    for r in rows:
        if r['group'] == 'hld':
            r['min_dist_to_training_nodes_vox'] = round(float(tree.query(nodes_in_band[r['patch']])[0].min()), 1)

    for name in os.listdir(out_dir):
        if name.startswith(('trn_', 'hld_')):
            shutil.rmtree(os.path.join(out_dir, name))
    for r in rows:
        if r['group'] in ('trn', 'hld'):
            src = os.path.join(a.repo, index[r['patch']]['tifxyz'])
            dst = os.path.join(out_dir, f'{r["group"]}_{r["patch"]}.tifxyz')
            os.makedirs(dst)
            for fname in ('meta.json', 'x.tif', 'y.tif', 'z.tif'):
                shutil.copy2(os.path.join(src, fname), dst)
    shutil.copy2(os.path.join(a.repo, 'LICENSE-DATA.md'), os.path.join(out_dir, '_LICENSE-DATA.md'))
    shutil.copy2(os.path.join(a.repo, 'CC-BY-NC-4.0-legalcode.txt'), os.path.join(out_dir, '_CC-BY-NC-4.0-legalcode.txt'))
    with open(os.path.join(out_dir, '_ATTRIBUTION.md'), 'w', encoding='utf-8') as f:
        f.write(
            '# Patch data: attribution and licence\n\n'
            'The `trn_*` and `hld_*` folders are the PHerc0813 tifxyz patches of '
            '[Ahggggg/first-letters-clean-sheets](https://github.com/Ahggggg/first-letters-clean-sheets) '
            f'(commit `{commit}`, `data/PHerc0813/`), traced by its authors with villa\'s `vc_grow_seg_from_seed` on the '
            'public m7 surface prediction of PHerc0813 volume 20250821151723 (Vesuvius Challenge data).\n\n'
            'Licence: CC BY-NC 4.0 (`_LICENSE-DATA.md`, full text `_CC-BY-NC-4.0-legalcode.txt`). '
            'Changes: none to the files; each folder was renamed with a `trn_` (training) or `hld_` (held-out) prefix '
            'for the A/B in PREREG_patches_ab.md, and patches with index.csv flags were left out. '
            'The authors state the patches are not shown to lie on one papyrus layer.\n')
    split = dict(
        source=f'https://github.com/Ahggggg/first-letters-clean-sheets/tree/{commit}/data/PHerc0813',
        licence='CC BY-NC 4.0', band=[a.z0, a.z1],
        rule=(f'excluded if index.csv flags non-empty or no node in [{a.z0}, {a.z1}); held-out (hld_) if the mean z '
              f'of the in-band nodes >= {a.split_z:g}, else training (trn_)'),
        training=trn, held_out=hld, excluded=[r['patch'] for r in rows if r['group'] == 'excluded'],
        counts=dict(trn=len(trn), hld=len(hld), excluded=sum(r['group'] == 'excluded' for r in rows)),
        area_in_band_cm2=dict(trn=round(sum(r['area_in_band_cm2'] for r in rows if r['group'] == 'trn'), 2),
                              hld=round(sum(r['area_in_band_cm2'] for r in rows if r['group'] == 'hld'), 2)),
        nodes_in_band=dict(trn=sum(r['nodes_in_band'] for r in rows if r['group'] == 'trn'),
                           hld=sum(r['nodes_in_band'] for r in rows if r['group'] == 'hld')),
        patches=rows,
    )
    with open(os.path.join(a.dataset, 'patches_split.json'), 'w') as f:
        json.dump(split, f, indent=1)
    print(json.dumps({k: split[k] for k in ('rule', 'counts', 'area_in_band_cm2', 'nodes_in_band', 'training',
                                            'held_out', 'excluded')}, indent=1))
    for r in rows:
        print(f'{r["group"]:8s} {r["patch"]:22s} in-band nodes {r["nodes_in_band"]:5d} area {r["area_in_band_cm2"]:5.2f} cm2 '
              f'z_mean {r["z_mean_in_band"]} spacing {r["node_spacing_vox"]} '
              + (f'min dist to trn {r["min_dist_to_training_nodes_vox"]}' if 'min_dist_to_training_nodes_vox' in r else ''))


if __name__ == '__main__':
    main()
