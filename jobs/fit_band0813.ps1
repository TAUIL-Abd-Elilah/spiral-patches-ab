# Patches A/B on PHerc0813 z 9000-10000 (PREREG_patches_ab.md): one spiral fit or one held-out eval pass, headless on
# Windows. Adapted copy of the First Letters fit_0813.ps1 recipe (villa 0fb38c45 + the same local patches: shell
# optional, FIT_SPIRAL_GPU_MEM_FRACTION, FIT_SPIRAL_CKPT_EVERY, crossing-cache check), with every path, cache and temp
# dir on C:/VesuviusTemp/band0813 and the settings in configs/*.json instead of an inline string.
#   train: arm A = tracks only (input_disable_patches); arm B = tracks + the trn_ patches as unverified patches.
#   eval:  resume the arm's checkpoint_fitted.ckpt with the same step count (0 iterations), load only the hld_
#          patches, skip the mesh export, and let the fitter write satisfied_fitted.json for them.
# usage: .\fit_band0813.ps1 -Arm A|B -Mode train|eval -Steps 20000 -Tag patchA [-ResumePath <ckpt>] [-Set 'key=json;key=json']
#        (-Set is one ';'-separated string: powershell -File does not split arrays; -DryRun stops before the fit)
param([ValidateSet('A', 'B')][string]$Arm, [ValidateSet('train', 'eval')][string]$Mode = 'train', [int]$Steps = 20000,
      [string]$Tag, [double]$MemFraction = 0.78, [int]$CkptEvery = 2000, [string]$ResumePath = '', [string]$Set = '',
      [switch]$DryRun)
$ErrorActionPreference = 'Stop'
$ROOT = 'C:/VesuviusTemp/band0813'
$DS = "$ROOT/dataset"
$SPIRAL = "$ROOT/villa_0fb/volume-cartographer/scripts/spiral"
$PY = 'C:/Users/PC/miniconda3/envs/vesuvius/python.exe'
$LOG = "$ROOT/logs/fit_$Tag.log"
if (-not $Tag) { throw 'no -Tag' }
if ($MemFraction -gt 0.80) { throw "MemFraction $MemFraction > 0.80" }
foreach ($d in @("$ROOT/out", "$ROOT/cache/spiral", "$ROOT/cache/torchinductor", "$ROOT/tmp", "$ROOT/logs")) {
    New-Item -ItemType Directory -Force $d | Out-Null }
Remove-Item "$ROOT/logs/fit_$Tag.done" -ErrorAction SilentlyContinue

# the outward sense comes from the catalog flags (dataset/catalog_orientation.json); never rewrite the spec here
$spec = Get-Content "$DS/spiral-scroll.json" -Raw | ConvertFrom-Json
$cat = Get-Content "$DS/catalog_orientation.json" -Raw | ConvertFrom-Json
if ($spec.spiral_outward_sense -ne $cat.spiral_outward_sense) { throw "spiral-scroll.json sense $($spec.spiral_outward_sense) != catalog $($cat.spiral_outward_sense)" }

$files = @("$ROOT/configs/common.json", "$ROOT/configs/arm_$Arm.json")
if ($Mode -eq 'eval') {
    $files += "$ROOT/configs/eval_hld.json"
    if (-not (Test-Path $ResumePath)) { throw "eval needs -ResumePath to the arm's checkpoint_fitted.ckpt ($ResumePath)" } }
$setArgs = @('--set', "optimizer_num_training_steps=$Steps")
foreach ($s in ($Set -split ';' | Where-Object { $_ })) { $setArgs += @('--set', $s) }
$overrides = & $PY -I "$ROOT/tools/make_overrides.py" $SPIRAL @files @setArgs
if ($LASTEXITCODE -ne 0) { throw 'config overrides failed validation' }
$env:FIT_SPIRAL_CONFIG_OVERRIDES = "$overrides"
$env:FIT_SPIRAL_OUT_DIR = "$ROOT/out"
$env:FIT_SPIRAL_CACHE_DIR = "$ROOT/cache/spiral"
$env:FIT_SPIRAL_RUN_TAG = $Tag
$env:WANDB_MODE = 'disabled'
$env:PYTHONPATH = 'D:/VesuviusCTCache/spiral/vcnative/pkg'   # MSVC build of vc.track_crossings (read-only)
$env:FIT_SPIRAL_GPU_MEM_FRACTION = "$MemFraction"   # stay out of WDDM shared memory
if ($Mode -eq 'train') {
    $env:FIT_SPIRAL_CKPT_EVERY = "$CkptEvery"
    Remove-Item Env:FIT_SPIRAL_SKIP_SAVE_MESH -ErrorAction SilentlyContinue
    if ($ResumePath) { $env:FIT_SPIRAL_RESUME_PATH = $ResumePath; $env:FIT_SPIRAL_RESUME_STEP = '0' }   # crash resume: step from the ckpt
    else { Remove-Item Env:FIT_SPIRAL_RESUME_PATH, Env:FIT_SPIRAL_RESUME_STEP -ErrorAction SilentlyContinue }
} else {
    $env:FIT_SPIRAL_CKPT_EVERY = '0'
    $env:FIT_SPIRAL_SKIP_SAVE_MESH = '1'          # held-out scoring only; the arm's own run exported the meshes
    $env:FIT_SPIRAL_RESUME_PATH = $ResumePath
    $env:FIT_SPIRAL_RESUME_STEP = "$Steps"         # fallback; the fitter takes completed_iterations from the ckpt
}
$env:TORCHINDUCTOR_CACHE_DIR = "$ROOT/cache/torchinductor"
$env:TMP = "$ROOT/tmp"; $env:TEMP = "$ROOT/tmp"
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'   # torch ships Intel libiomp5md, the module LLVM libomp
$env:PYTHONIOENCODING = 'utf-8'

$ErrorActionPreference = 'Continue'
$manifest = [ordered]@{ tag = $Tag; arm = $Arm; mode = $Mode; steps = $Steps; mem_fraction = $MemFraction
    ckpt_every = $env:FIT_SPIRAL_CKPT_EVERY; resume_path = $ResumePath; overrides = ($overrides | ConvertFrom-Json)
    villa_head = (& git -C "$ROOT/villa_0fb" rev-parse HEAD); villa_local_diff_lines = ((& git -C "$ROOT/villa_0fb" diff) | Measure-Object -Line).Lines
    started = (Get-Date -Format s) }
$manifest | ConvertTo-Json -Depth 5 | Set-Content -Encoding ascii "$ROOT/logs/fit_$Tag.manifest.json"

if ($DryRun) {
    Get-ChildItem Env:FIT_SPIRAL_*, Env:TORCHINDUCTOR_CACHE_DIR, Env:PYTHONPATH, Env:TMP | ForEach-Object { "$($_.Name)=$($_.Value)" }
    Write-Host "DRY_RUN $Tag (no fit started)"; exit 0 }
Set-Location $SPIRAL
# cmd redirection keeps the log plain UTF-8 (Tee-Object writes UTF-16 and wraps stderr lines as PowerShell errors).
# No quotes on purpose: none of these paths has a space, and cmd /c strips the first and last quote of a line with more
# than two. Train and eval pass the same --dataset string, which the checkpoint check compares.
cmd /c "$($PY -replace '/', '\') -u fit_spiral.py --dataset $($DS -replace '/', '\') > $($LOG -replace '/', '\') 2>&1"
$rc = $LASTEXITCODE
$run = Get-ChildItem "$ROOT/out" -Directory -Filter "*_slice-9000-10000_*-patch_$Tag" | Sort-Object LastWriteTime | Select-Object -Last 1
$runPath = if ($run) { $run.FullName -replace '\\', '/' } else { '' }
if ($rc -eq 0 -and $run -and (Test-Path "$runPath/checkpoint_fitted.ckpt")) {
    if ($Mode -eq 'train') { Remove-Item "$runPath/checkpoint_periodic.ckpt" -ErrorAction SilentlyContinue }   # crash-resume copy only
    else { Remove-Item "$runPath/checkpoint_fitted.ckpt" }    # a copy of the arm's own checkpoint (0 iterations run)
}
Set-Content -Path "$ROOT/logs/fit_$Tag.done" -Value @("FIT_DONE $Tag rc=$rc", $runPath) -Encoding ascii
Write-Host "FIT_DONE $Tag rc=$rc $runPath"
exit $rc
