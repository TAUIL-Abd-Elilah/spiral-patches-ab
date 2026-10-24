#!/bin/bash
# job_patches_ab_smoke.sh (23 Oct): GPU-queue smoke test of the patches A/B on PHerc0813 z 9000-10000
# (PREREG_patches_ab.md). Arm A and arm B for 200 steps each (track/patch DT switched on from step 50, so the DT code
# paths run before the 20k fits reach step 10000), then the held-out eval pass on each 200-step checkpoint, then log
# checks. Runner contract: one line in a serial GPU queue file, output in the queue log.
# Writes logs/smoke.ok on success (job_patches_ab_arm.sh refuses to start without it); exit 0 = SMOKE_OK.
set -u
R=C:/VesuviusTemp/band0813; L=$R/logs; PS1W=$(cygpath -w $R/fit_band0813.ps1)
SET='loss_start_track_dt=50;loss_start_patch_dt=50;loss_start_unverified_patch_dt=50'
mkdir -p $L; rm -f $L/smoke.ok
# Data lives on D: since 23 Oct (C:/VesuviusTemp/band0813 is a junction to D:/PrizeTemp/band0813).
avail_gb() { df -BG --output=avail /d | tail -1 | tr -dc 0-9; }
c_gb() { df -BG --output=avail /c | tail -1 | tr -dc 0-9; }
[ "$(c_gb)" -ge 3 ] || { echo "C: has $(c_gb) GB free; need >= 3"; exit 1; }
[ "$(avail_gb)" -ge 11 ] || { echo "D: has $(avail_gb) GB free; need >= 11 (8 GB floor + ~2.5 GB smoke peak)"; exit 1; }
fit() { local tag=$1; shift; powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$PS1W" -Tag $tag "$@" > $L/ps1_$tag.out 2>&1; }
run_dir() { sed -n 2p $L/fit_$1.done 2>/dev/null | tr -d '\r'; }
rc_of() { sed -n 1p $L/fit_$1.done 2>/dev/null | tr -d '\r' | sed 's/.*rc=//'; }

# GPU memory sampler: dedicated (nvidia-smi) and WDDM shared usage of our fit process (command line has band0813)
gpu_watch() {
  while true; do
    ded=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' \r')
    sh=$(powershell -NoProfile -Command "\$p=(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'fit_spiral' -and \$_.CommandLine -match 'band0813' }).ProcessId | Select-Object -First 1; if (\$p) { \$s=(Get-Counter \"\GPU Process Memory(pid_\${p}_*)\Shared Usage\" -ErrorAction SilentlyContinue).CounterSamples | Measure-Object CookedValue -Sum; [int](\$s.Sum/1MB) } else { -1 }" | tr -d '\r')
    echo "$(date +%H:%M:%S) dedicated_MiB $ded shared_MB $sh"; sleep 60
  done
}
gpu_watch > $L/gpu_smoke.log 2>&1 & WATCH=$!

echo "$(date +%H:%M) smoke A"; fit smkA -Arm A -Mode train -Steps 200 -CkptEvery 0 -Set "$SET"
echo "$(date +%H:%M) smoke B"; fit smkB -Arm B -Mode train -Steps 200 -CkptEvery 0 -Set "$SET"
for a in A B; do
  ck="$(run_dir smk$a)/checkpoint_fitted.ckpt"
  echo "$(date +%H:%M) eval $a from $ck"
  [ -f "$ck" ] && fit smk${a}_eval -Arm $a -Mode eval -Steps 200 -ResumePath "$ck" -Set "$SET"
done
kill $WATCH 2>/dev/null

fail=0
check() { if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fail=1; fi; }
pool() { grep -o 'lasagna normals: resident pool [0-9,]*' $L/fit_$1.log 2>/dev/null | tail -1 | grep -o '[0-9,]*$' | tr -d ,; }
sat() { grep -o '^satisfied_tracks = [0-9]*/[0-9]*' $L/fit_$1.log 2>/dev/null | tail -1; }
for t in smkA smkB; do
  check "$t rc=0" '[ "$(rc_of $t)" = 0 ]'
  check "$t lasagna normals pool > 1000 bricks (got $(pool $t))" '[ "$(pool $t)" -gt 1000 ] 2>/dev/null'
done
for t in smkA smkB smkA_eval smkB_eval; do
  check "$t loaded 1270348 tracks" 'grep -q "loaded 1270348 tracks within z-roi \[9000, 10000)" $L/fit_$t.log'
done
check "smkA loads no patches" 'grep -q "skipping all verified/unverified patch loading" $L/fit_smkA.log'
check "smkB loads 13 trn_ patches, 0 verified" 'grep -q "^ loaded 13 unverified patches" $L/fit_smkB.log && grep -q "^ loaded 0 patches" $L/fit_smkB.log'
check "smkB keeps all 13 (no trusted geometry to mask against)" 'grep -q "dropped 0 fully-masked patches; 13 remain" $L/fit_smkB.log'
for a in A B; do
  t=smk${a}_eval
  check "$t rc=0" '[ "$(rc_of $t)" = 0 ]'
  check "$t resumed at 200 and ran 0 steps" 'grep -q "resuming from .* at iteration 200" $L/fit_$t.log && ! grep -q "^step [0-9]*:" $L/fit_$t.log'
  check "$t loads the 7 hld_ patches and scores them" 'grep -q "^ loaded 7 unverified patches" $L/fit_$t.log && grep -q "^unverified_satisfied_patches = [0-9]*/7 " $L/fit_$t.log'
  check "$t satisfied_tracks equals the 200-step run ($(sat $t) vs $(sat smk$a))" '[ -n "$(sat $t)" ] && [ "$(sat $t)" = "$(sat smk$a)" ]'
done
maxsh=$(grep -o 'shared_MB [0-9-]*' $L/gpu_smoke.log | awk '{print $2}' | sort -n | tail -1)
maxded=$(grep -o 'dedicated_MiB [0-9]*' $L/gpu_smoke.log | awk '{print $2}' | sort -n | tail -1)
check "WDDM shared GPU memory peak <= 300 MB (got ${maxsh:-?} MB; dedicated peak ${maxded:-?} MiB)" '[ "${maxsh:-0}" -le 300 ]'
echo "--- log excerpts"
grep -h -E "^step 0:|^satisfied_tracks|^unverified_satisfied|resident pool|^patch filter regex|unverified patches: masked|resuming from|Traceback|Error" $L/fit_smk*.log | cut -c1-300
# clean up the smoke runs (checkpoints, meshes); keep logs and satisfied_*.json
for t in smkA smkB smkA_eval smkB_eval; do
  d=$(run_dir $t)
  if [ -n "$d" ] && [ -d "$d" ]; then cp "$d/satisfied_fitted.json" $L/satisfied_$t.json 2>/dev/null; rm -rf "$d"; fi
done
echo "D: free after clean-up: $(avail_gb) GB"
if [ $fail = 0 ]; then date > $L/smoke.ok; echo SMOKE_OK; else echo SMOKE_FAILED; fi
exit $fail
