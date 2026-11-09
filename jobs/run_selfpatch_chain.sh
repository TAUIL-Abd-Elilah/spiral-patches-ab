#!/bin/bash
# run_selfpatch_chain.sh (9 Nov): S1, R1, S2, R2 in the pre-registered order, then scoring; detached, beside the
# First Letters queue job (fit peak ~5 GB + FL ~3.4 GB << 20 GB). Each step logs to logs/chain_<tag>.out.
R=C:/VesuviusTemp/band0813; cd $R
for j in "S 1" "R 1" "S 2" "R 2"; do
  set -- $j; t=self$1; [ $2 = 1 ] || t=self$1_s$2
  echo "$(date +%F_%T) start $t" >> logs/chain.log
  bash job_selfpatch_arm.sh $1 $2 > logs/chain_$t.out 2>&1; echo "$(date +%F_%T) end $t rc=$? $(tail -1 logs/chain_$t.out)" >> logs/chain.log
  grep -q "^ARM_OK" logs/chain_$t.out || { echo "$(date +%F_%T) CHAIN_STOPPED at $t" >> logs/chain.log; exit 1; }
  if [ $1 = R ]; then bash score_selfpatch.sh $2 > logs/score_seed$2.out 2>&1; echo "$(date +%F_%T) scored seed $2 rc=$?" >> logs/chain.log; fi
done
echo "$(date +%F_%T) CHAIN_DONE" >> logs/chain.log
