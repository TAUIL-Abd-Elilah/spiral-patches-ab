#!/bin/bash
# score_selfpatch.sh <seed 1|2> (9 Nov): PREREG_selfpatch.md scoring with score_ab.py unchanged. score_ab.py names its
# arms A and B and reports B_minus_A, so each comparison passes the baseline as "A" and the tested arm as "B":
#   S-A (primary), R-A, S-R. S-A passes --go-within2-points 3 (the self-patch bar); the seed-2 sign rule is read by hand.
set -eu
R=C:/VesuviusTemp/band0813; PY=C:/Users/PC/miniconda3/envs/vesuvius/python.exe; S=$1
sfx=""; [ $S = 1 ] || sfx=_s$S
m() { d=$(sed -n 2p $R/logs/fit_$1.done | tr -d '\r'); echo "$d/meshes/fitted_$1"; }
A=patchA$sfx; SS=selfS$sfx; RR=selfR$sfx
P="--patches $R/dataset/unverified_patches --prefix hld_ --z0 9000 --z1 10000 --edge 20 --boot 10000 --seed 12345"
cd $R
$PY score_ab.py --arm A=$(m $A) --arm B=$(m $SS) $P --log A=logs/fit_$A.log --log B=logs/fit_$SS.log --go-within2-points 3 --out results/selfpatch_S_vs_A$sfx.json
[ -f logs/fit_$RR.done ] && $PY score_ab.py --arm A=$(m $A) --arm B=$(m $RR) $P --out results/selfpatch_R_vs_A$sfx.json
[ -f logs/fit_$RR.done ] && $PY score_ab.py --arm A=$(m $RR) --arm B=$(m $SS) $P --out results/selfpatch_S_vs_R$sfx.json
echo SCORED seed $S
