"""fit_band0813.py (9 Nov 2026): Python port of fit_band0813.ps1, line for line, because Windows PowerShell 5.1 on this
PC hangs at startup (PREREG_selfpatch.md amendment 1). Same environment, same overrides, same fitter call, same
manifest, same checkpoint clean-up, same logs/fit_<tag>.done.

usage: fit_band0813.py --arm A|B|S|R --mode train|eval --steps 20000 --tag T --mem-fraction 0.78 --ckpt-every 2000
       [--resume-path P] [--set "k=v;k=v"] [--dry-run]
"""
import argparse, datetime, glob, json, os, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument('--arm', choices=['A', 'B', 'S', 'R'], required=True)
ap.add_argument('--mode', choices=['train', 'eval'], default='train')
ap.add_argument('--steps', type=int, default=20000); ap.add_argument('--tag', required=True)
ap.add_argument('--mem-fraction', type=float, default=0.78); ap.add_argument('--ckpt-every', type=int, default=2000)
ap.add_argument('--resume-path', default=''); ap.add_argument('--set', default=''); ap.add_argument('--dry-run', action='store_true')
a = ap.parse_args()

ROOT = 'C:/VesuviusTemp/band0813'
DS = f'{ROOT}/dataset'
SPIRAL = f'{ROOT}/villa_0fb/volume-cartographer/scripts/spiral'
PY = 'C:/Users/PC/miniconda3/envs/vesuvius/python.exe'
LOG = f'{ROOT}/logs/fit_{a.tag}.log'
DONE = f'{ROOT}/logs/fit_{a.tag}.done'
if a.mem_fraction > 0.80:
    sys.exit(f'MemFraction {a.mem_fraction} > 0.80')
for d in (f'{ROOT}/out', f'{ROOT}/cache/spiral', f'{ROOT}/cache/torchinductor', f'{ROOT}/tmp', f'{ROOT}/logs'):
    os.makedirs(d, exist_ok=True)
if os.path.exists(DONE):
    os.remove(DONE)

spec = json.load(open(f'{DS}/spiral-scroll.json')); cat = json.load(open(f'{DS}/catalog_orientation.json'))
if spec['spiral_outward_sense'] != cat['spiral_outward_sense']:
    sys.exit(f"spiral-scroll.json sense {spec['spiral_outward_sense']} != catalog {cat['spiral_outward_sense']}")

files = [f'{ROOT}/configs/common.json', f'{ROOT}/configs/arm_{a.arm}.json']
if a.mode == 'eval':
    files.append(f'{ROOT}/configs/eval_hld.json')
    if not os.path.exists(a.resume_path):
        sys.exit(f"eval needs --resume-path to the arm's checkpoint_fitted.ckpt ({a.resume_path})")
set_args = ['--set', f'optimizer_num_training_steps={a.steps}']
for s in [s for s in a.set.split(';') if s]:
    set_args += ['--set', s]
r = subprocess.run([PY, '-I', f'{ROOT}/tools/make_overrides.py', SPIRAL, *files, *set_args], capture_output=True, text=True)
if r.returncode:
    sys.exit('config overrides failed validation\n' + r.stderr)
overrides = r.stdout.strip()

env = dict(os.environ)
env.update(FIT_SPIRAL_CONFIG_OVERRIDES=overrides, FIT_SPIRAL_OUT_DIR=f'{ROOT}/out', FIT_SPIRAL_CACHE_DIR=f'{ROOT}/cache/spiral',
           FIT_SPIRAL_RUN_TAG=a.tag, WANDB_MODE='disabled', PYTHONPATH='D:/VesuviusCTCache/spiral/vcnative/pkg',
           FIT_SPIRAL_GPU_MEM_FRACTION=str(a.mem_fraction))
if a.mode == 'train':
    env['FIT_SPIRAL_CKPT_EVERY'] = str(a.ckpt_every)
    env.pop('FIT_SPIRAL_SKIP_SAVE_MESH', None)
    if a.resume_path:
        env.update(FIT_SPIRAL_RESUME_PATH=a.resume_path, FIT_SPIRAL_RESUME_STEP='0')
    else:
        env.pop('FIT_SPIRAL_RESUME_PATH', None); env.pop('FIT_SPIRAL_RESUME_STEP', None)
else:
    env.update(FIT_SPIRAL_CKPT_EVERY='0', FIT_SPIRAL_SKIP_SAVE_MESH='1', FIT_SPIRAL_RESUME_PATH=a.resume_path,
               FIT_SPIRAL_RESUME_STEP=str(a.steps))
env.update(TORCHINDUCTOR_CACHE_DIR=f'{ROOT}/cache/torchinductor', TMP=f'{ROOT}/tmp', TEMP=f'{ROOT}/tmp',
           KMP_DUPLICATE_LIB_OK='TRUE', PYTHONIOENCODING='utf-8')


def git(*args):
    return subprocess.run(['git', '-C', f'{ROOT}/villa_0fb', *args], capture_output=True, text=True).stdout


manifest = dict(tag=a.tag, arm=a.arm, mode=a.mode, steps=a.steps, mem_fraction=a.mem_fraction,
                ckpt_every=env['FIT_SPIRAL_CKPT_EVERY'], resume_path=a.resume_path, overrides=json.loads(overrides),
                villa_head=git('rev-parse', 'HEAD').strip(), villa_local_diff_lines=len(git('diff').splitlines()),
                started=datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), launcher='fit_band0813.py')
json.dump(manifest, open(f'{ROOT}/logs/fit_{a.tag}.manifest.json', 'w'), indent=2)

if a.dry_run:
    for k in sorted(env):
        if k.startswith('FIT_SPIRAL_') or k in ('TORCHINDUCTOR_CACHE_DIR', 'PYTHONPATH', 'TMP'):
            print(f'{k}={env[k]}')
    print(f'DRY_RUN {a.tag} (no fit started)'); sys.exit(0)

with open(LOG, 'w', encoding='utf-8') as lf:
    rc = subprocess.run([PY, '-u', 'fit_spiral.py', '--dataset', DS.replace('/', '\\')], cwd=SPIRAL, env=env,
                        stdout=lf, stderr=subprocess.STDOUT).returncode
runs = sorted(glob.glob(f'{ROOT}/out/*_slice-9000-10000_*-patch_{a.tag}'), key=os.path.getmtime)
run = runs[-1].replace('\\', '/') if runs else ''
if rc == 0 and run and os.path.exists(f'{run}/checkpoint_fitted.ckpt'):
    if a.mode == 'train':
        try: os.remove(f'{run}/checkpoint_periodic.ckpt')
        except FileNotFoundError: pass
    else:
        os.remove(f'{run}/checkpoint_fitted.ckpt')
open(DONE, 'w').write(f'FIT_DONE {a.tag} rc={rc}\n{run}\n')
print(f'FIT_DONE {a.tag} rc={rc} {run}')
sys.exit(rc)
