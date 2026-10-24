"""CPU dry run of the patches A/B inputs with villa 0fb38c45's own code, no GPU (CUDA hidden), no fit:
dataset resolution, the pre-GPU session validation for each arm and eval pass, the scroll spec and umbilicus, and
the patch loading/ROI filtering each arm will do (regex, erosion, z-ROI). The track count is checked separately by
verify_band_tracks.py and the normals pool by pack_lasagna.py.

usage: dry_run.py <band root>     (C:/VesuviusTemp/band0813)
"""
import json
import os
import re
import subprocess
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # '' would unset it on Windows


def cli_tolerated(error, cfg):
    """validate_session_request is the service's check: it also flags optional inputs that are absent from this
    dataset. The headless CLI skips them (no fibers or PCL files: warnings; no outer_shell: the FL shell-optional
    patch, with the shell weights at 0; grad_mag and surf_sdt: not opened in grad_mag mode with
    loss_weight_dense_spacing 0). Anything else blocks the fit."""
    field, msg = error['field'], error['message']
    if field in ('fibers', 'outer_shell') and msg == 'Path is not a directory':
        return cfg['loss_weight_shell_outer'] == 0 and cfg['loss_weight_shell_patch_radius'] == 0 or field == 'fibers'
    if field.startswith('pcls[') and msg.startswith('PCL file does not exist'):
        return True
    if field in ('gradient_magnitude', 'surf_sdt') and msg == 'Path is not a directory':
        return cfg['dense_spacing_mode'] == 'grad_mag' and cfg['loss_weight_dense_spacing'] == 0
    return False


def main():
    root = sys.argv[1]
    spiral = os.path.join(root, 'villa_0fb/volume-cartographer/scripts/spiral')
    dataset = os.path.join(root, 'dataset')
    sys.path.insert(0, spiral)
    import torch
    assert not torch.cuda.is_available()
    from fit_session import (SpiralRunConfig, conventional_input_paths, load_scroll_spec, resolve_dataset_root,
                             validate_session_request)
    from spiral_helpers import erode_patch_valid_region, patch_intersects_z_roi
    from tifxyz import load_tifxyz
    from umbilicus import json_umbilicus_z_to_yx

    report = {}
    res = resolve_dataset_root(dataset)
    report['resolution'] = dict(resolved=res.resolved, missing_required=res.missing_required,
                                missing_optional=res.missing_optional, warnings=res.warnings)
    spec = load_scroll_spec(dataset)
    report['scroll_spec'] = spec.manifest()
    paths = conventional_input_paths(dataset, spec, output_directory=os.path.join(root, 'out'),
                                     cache_directory=os.path.join(root, 'cache/spiral'))

    def merged(*names, **sets):
        cmd = [sys.executable, '-I', os.path.join(root, 'tools/make_overrides.py'), spiral] + \
              [os.path.join(root, 'configs', n) for n in names]
        for k, v in sets.items():
            cmd += ['--set', f'{k}={json.dumps(v)}']
        return json.loads(subprocess.check_output(cmd, text=True))

    runs = {
        'A_train': merged('common.json', 'arm_A.json'),
        'B_train': merged('common.json', 'arm_B.json'),
        'A_eval': merged('common.json', 'arm_A.json', 'eval_hld.json'),
        'B_eval': merged('common.json', 'arm_B.json', 'eval_hld.json'),
    }
    report['validate_session_request'] = {}
    report['validate_session_request_blocking'] = {}
    report['patches'] = {}
    for name, cfg in runs.items():
        run = SpiralRunConfig(z_begin=cfg['z_begin'], z_end=cfg['z_end'], config=cfg)
        errors = validate_session_request(paths, run)
        report['validate_session_request'][name] = errors
        report['validate_session_request_blocking'][name] = [e for e in errors if not cli_tolerated(e, cfg)]
        # the fitter's patch loading (fit_spiral._load_patches_from_dir + the ROI filter of load_host_inputs)
        if cfg.get('input_disable_patches'):
            report['patches'][name] = 'patches disabled: "skipping all verified/unverified patch loading"'
            continue
        out = {}
        for label, pdir in (('verified', paths.verified_patches), ('unverified', paths.unverified_patches)):
            entries = sorted(os.listdir(pdir))
            rx = cfg.get('patch_uuid_filter_regex')
            kept = [e for e in entries if rx is None or re.search(rx, e)]
            loaded, survived, nodes = [], [], 0
            for e in kept:
                p = load_tifxyz(os.path.join(pdir, e))
                loaded.append(e)
                cells = p.erosion_cells(cfg.get('patch_erode_patches', 1))
                if cells > 0 and not erode_patch_valid_region(p, cells):
                    continue
                if patch_intersects_z_roi(p, cfg['z_begin'], cfg['z_end']):
                    survived.append(e)
                    nodes += int(p.valid_vertex_mask.sum())
            out[label] = dict(entries=len(entries), regex=rx, kept=len(kept), loaded=len(loaded),
                              in_roi_after_erosion=len(survived), valid_nodes_after_erosion=nodes, names=survived)
        report['patches'][name] = out
    zs = list(range(cfg['z_begin'], cfg['z_end'], 100))
    u = json_umbilicus_z_to_yx(paths.umbilicus)
    pts = json.load(open(paths.umbilicus))['control_points']
    in_band = [p for p in pts if cfg['z_begin'] - 200 <= p['z'] <= cfg['z_end'] + 200]
    report['umbilicus'] = dict(path=paths.umbilicus, control_points=len(pts),
                               control_points_within_200_of_band=len(in_band),
                               yx_at=[[z, [round(float(v), 1) for v in u(z)]] for z in zs[::3]])
    ok = (not res.missing_required and all(not v for v in report['validate_session_request_blocking'].values())
          and report['patches']['B_train']['unverified']['in_roi_after_erosion'] == 13
          and report['patches']['B_eval']['unverified']['in_roi_after_erosion'] == 7
          and report['patches']['B_train']['verified']['loaded'] == 0
          and spec.spiral_outward_sense == 'CW')
    report['DRY_RUN_OK'] = bool(ok)
    with open(os.path.join(root, 'logs', 'dry_run.json'), 'w') as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
