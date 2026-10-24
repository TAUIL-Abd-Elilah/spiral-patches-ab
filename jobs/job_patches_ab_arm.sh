#!/bin/bash
# job_patches_ab_arm.sh <A|B> [seed] (23 Oct): one arm of the patches A/B on PHerc0813 z 9000-10000
# (PREREG_patches_ab.md) on the GPU queue: the 20k-step fit (fit_band0813.ps1, ckpt every 2000, GPU fraction 0.78),
# then the held-out eval pass on its checkpoint_fitted.ckpt, then log checks. Needs logs/smoke.ok from
# job_patches_ab_smoke.sh. Tags: patch<arm> (seed 1) or patch<arm>_s<seed>. ~3.2 h fit + ~10 min eval.
set -u
ARM=$1; SEED=${2:-1}
R=C:/VesuviusTemp/band0813; L=$R/logs; PS1W=$(cygpath -w $R/fit_band0813.ps1)
case $ARM in A|B) ;; *) echo "arm must be A or B"; exit 2;; esac
TAG=patch$ARM; [ "$SEED" = 1 ] || TAG=patch${ARM}_s$SEED
SET=""; [ "$SEED" = 1 ] || SET="optimizer_random_seed=$SEED"
[ -f $L/smoke.ok ] || { echo "no logs/smoke.ok: run job_patches_ab_smoke.sh first (or it failed)"; exit 1; }
# Data lives on D: since 23 Oct (C:/VesuviusTemp/band0813 is a junction to D:/PrizeTemp/band0813).
avail_gb() { df -BG --output=avail /d | tail -1 | tr -dc 0-9; }
c_gb() { df -BG --output=avail /c | tail -1 | tr -dc 0-9; }
[ "$(c_gb)" -ge 3 ] || { echo "C: has $(c_gb) GB free; need >= 3"; exit 1; }
[ "$(avail_gb)" -ge 10 ] || { echo "D: has $(avail_gb) GB free; need >= 10 (8 GB floor + ~1.6 GB fit peak)"; exit 1; }
fit() { local tag=$1; shift; powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$PS1W" -Tag $tag "$@" > $L/ps1_$tag.out 2>&1; }
run_dir() { sed -n 2p $L/fit_$1.done 2>/dev/null | tr -d '\r'; }
rc_of() { sed -n 1p $L/fit_$1.done 2>/dev/null | tr -d '\r' | sed 's/.*rc=//'; }
gpu_watch() {
  while true; do
    ded=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' \r')
    sh=$(powershell -NoProfile -Command "\$p=(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'fit_spiral' -and \$_.CommandLine -match 'band0813' }).ProcessId | Select-Object -First 1; if (\$p) { \$s=(Get-Counter \"\GPU Process Memory(pid_\${p}_*)\Shared Usage\" -ErrorAction SilentlyContinue).CounterSamples | Measure-Object CookedValue -Sum; [int](\$s.Sum/1MB) } else { -1 }" | tr -d '\r')
    echo "$(date +%H:%M:%S) dedicated_MiB $ded shared_MB $sh"; sleep 120
  done
}
gpu_watch > $L/gpu_$TAG.log 2>&1 & WATCH=$!

echo "$(date +%H:%M) fit $TAG (arm $ARM, seed $SEED)"
fit $TAG -Arm $ARM -Mode train -Steps 20000 -MemFraction 0.78 -CkptEvery 2000 -Set "$SET"
ck="$(run_dir $TAG)/checkpoint_fitted.ckpt"
echo "$(date +%H:%M) fit rc=$(rc_of $TAG) -> $(run_dir $TAG)"
if [ "$(rc_of $TAG)" = 0 ] && [ -f "$ck" ]; then
  echo "$(date +%H:%M) eval ${TAG}_eval from $ck"
  fit ${TAG}_eval -Arm $ARM -Mode eval -Steps 20000 -MemFraction 0.78 -ResumePath "$ck" -Set "$SET"
fi
kill $WATCH 2>/dev/null

fail=0
check() { if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fail=1; fi; }
pool() { grep -o 'lasagna normals: resident pool [0-9,]*' $L/fit_$1.log 2>/dev/null | tail -1 | grep -o '[0-9,]*$' | tr -d ,; }
sat() { grep -o '^satisfied_tracks = [0-9]*/[0-9]*' $L/fit_$1.log 2>/dev/null | tail -1; }
satn() { sat $1 | grep -o '= [0-9]*' | tr -dc 0-9; }
# Amendment 1 (24 Oct): GPU float non-determinism flips a few tracks between the fit and its re-scoring (smoke: 55 and
# 186 of ~96k), so the eval must agree with the fit to within 0.5 % of the fit's satisfied tracks, not exactly.
close() { [ -n "$(satn $1)" ] && [ -n "$(satn $2)" ] && awk -v a="$(satn $1)" -v b="$(satn $2)" 'BEGIN{d=a-b; if(d<0)d=-d; exit !(d<=0.005*b)}'; }
check "$TAG rc=0" '[ "$(rc_of $TAG)" = 0 ]'
check "$TAG lasagna normals pool > 1000 bricks (got $(pool $TAG))" '[ "$(pool $TAG)" -gt 1000 ] 2>/dev/null'
check "$TAG loaded 1270348 tracks" 'grep -q "loaded 1270348 tracks within z-roi \[9000, 10000)" $L/fit_$TAG.log'
if [ $ARM = A ]; then check "$TAG loads no patches" 'grep -q "skipping all verified/unverified patch loading" $L/fit_$TAG.log'
else check "$TAG loads 13 trn_ patches, 0 verified" 'grep -q "^ loaded 13 unverified patches" $L/fit_$TAG.log && grep -q "^ loaded 0 patches" $L/fit_$TAG.log'; fi
check "${TAG}_eval rc=0" '[ "$(rc_of ${TAG}_eval)" = 0 ]'
check "${TAG}_eval resumed at 20000 and ran 0 steps" 'grep -q "resuming from .* at iteration 20000" $L/fit_${TAG}_eval.log && ! grep -q "^step [0-9]*:" $L/fit_${TAG}_eval.log'
check "${TAG}_eval scores the 7 hld_ patches" 'grep -q "^unverified_satisfied_patches = [0-9]*/7 " $L/fit_${TAG}_eval.log'
check "${TAG}_eval satisfied_tracks within 0.5 % of the fit ($(sat ${TAG}_eval) vs $(sat $TAG))" 'close ${TAG}_eval $TAG'
maxsh=$(grep -o 'shared_MB [0-9-]*' $L/gpu_$TAG.log | awk '{print $2}' | sort -n | tail -1)
maxded=$(grep -o 'dedicated_MiB [0-9]*' $L/gpu_$TAG.log | awk '{print $2}' | sort -n | tail -1)
check "WDDM shared GPU memory peak <= 300 MB (got ${maxsh:-?} MB; dedicated peak ${maxded:-?} MiB)" '[ "${maxsh:-0}" -le 300 ]'
d=$(run_dir ${TAG}_eval); [ -n "$d" ] && cp "$d/satisfied_fitted.json" $L/satisfied_${TAG}_eval.json 2>/dev/null
grep -h -E "^satisfied_tracks|^unverified_satisfied|resident pool|Traceback|Error" $L/fit_$TAG.log $L/fit_${TAG}_eval.log | cut -c1-300
echo "meshes: $(run_dir $TAG)/meshes/fitted_$TAG ; D: free $(avail_gb) GB"
[ $fail = 0 ] && echo "ARM_OK $TAG" || echo "ARM_FAILED $TAG"
exit $fail
