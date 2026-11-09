#!/bin/bash
# rebuild_inputs.sh (9 Nov): the band's tracks DBM and Lasagna nx/ny mirror were deleted from C: between 26 Oct and 9 Nov
# (not by this harness). Rebuild both with the harness's own tools and the same arguments as on 23 Oct, check the cut
# against the 23 Oct summary (tracks, points, source-id sha1), then start the self-patch chain.
R=C:/VesuviusTemp/band0813; PY=C:/Users/PC/miniconda3/envs/vesuvius/python.exe; L=$R/logs; cd $R
echo "$(date +%F_%T) rebuild start" >> $L/chain.log
mkdir -p dataset/tracks
$PY -I -u tools/cut_band_tracks.py https://dl.ash2txt.org/datasets/spiral_datasets/PHerc0813/20250821151723/tracks/ \
  PHerc0813_20250821151723_surface_m7_L0_th0.2.dbm dataset/tracks/PHerc0813_20250821151723_surface_m7_L0_th0.2_z9000-10000.dbm \
  --z0 9000 --z1 10000 --cache cache/tracks_index > $L/cut_band2.out 2> $L/cut_band2.err
rc=$?; echo "$(date +%F_%T) cut rc=$rc" >> $L/chain.log
grep -q '"source_ids_sha1": "efe3a8f02883cb62c83f91f97743b4fbd77fe3a9"' $L/cut_band2.out && grep -q '1,270,348 tracks, 47,877,707 points' $L/cut_band2.out \
  || { echo "$(date +%F_%T) REBUILD_FAILED cut does not match 23 Oct" >> $L/chain.log; exit 1; }
for c in nx ny; do
  $PY -I -u tools/mirror_zarr_fast.py PHerc0813/representations/predictions/lasagna/20250821151723-lasagna-20260419180421/PHerc0813_$c.ome.zarr 2 \
    dataset/lasagna_inputs/las_008_$c.ome.zarr --z0 2234 --z1 2516 --workers 4 >> $L/mirror_lasagna2.out 2>&1 \
    || { echo "$(date +%F_%T) REBUILD_FAILED mirror $c" >> $L/chain.log; exit 1; }
done
echo "$(date +%F_%T) rebuild ok; $(tail -1 $L/mirror_lasagna2.out)" >> $L/chain.log
bash run_selfpatch_chain.sh
