# Result: the spiral fit's own "good" regions as patch inputs to a refit (PHerc0813 z 9000-10000)

Pre-registration: [PREREG_selfpatch.md](PREREG_selfpatch.md), pushed before any S or R fit (d0c2385; amendment 1,
dbe9760, before any fit step). All four runs ran on 9 Nov 2026, one RTX 3090, about 70 minutes each including the
eval pass. All four are valid on every pre-registered check:

- rc 0;
- 1,270,348 tracks;
- normals pool 9,762 bricks;
- 46 `slf_` / `rnd_` patches loaded, none masked, 0 verified;
- eval resumed at 20,000 with 0 steps;
- eval satisfied tracks within 0.5 %;
- shared GPU memory 86 MB.

**Decision: NO-GO by the pre-registered bar.** Feeding back regions where two seeds agree and one winding holds the
sheet moves the refit closer to the held-out human patches by +1.9 points (seed 1) and +2.5 points (seed 2). The bar
was +3.0 with the 95 % CI above 0. Feeding back the same number of random tiles from the same fit does not help:
-1.5 and -0.9 points. The selector beats the random tiles by +3.4 points on both seeds.

## Numbers

Share of held-out patch nodes within 2 voxels of a fitted winding: 7 held-out patches, z 9020-9980, paired bootstrap
over patches (10,000 draws, seed 12345), `score_ab.py` unchanged.

| comparison | seed 1 | seed 2 |
|---|---|---|
| **S - A** (primary) | **+1.9 [-1.1, +4.4]** (32.3 → 34.1 %) | +2.5 [-0.9, +5.0] (32.4 → 35.0 %) |
| R - A | -1.5 [-2.8, +0.3] (32.3 → 30.8 %) | -0.9 [-4.4, +1.9] (32.4 → 31.5 %) |
| S - R (does the selection matter?) | +3.4 [-0.1, +6.2] | +3.4 [+0.3, +7.4] |

| other measures | A s1 / s2 | S s1 / s2 | R s1 / s2 |
|---|---|---|---|
| within 4 voxels | 57.9 / 58.5 % | 60.3 / 60.1 % | 57.3 / 57.0 % |
| duplicate-sheet share | 21.3 / 21.2 % | 21.1 / 21.0 % | 19.5 / 19.5 % |
| winding hops in held-out patches | 1,088 / 1,116 | 1,167 / 1,102 | 1,020 / 1,025 |
| satisfied tracks | 10.5 / 10.5 % | 10.6 / 10.6 % | 10.7 / 10.7 % |
| fitter's own held-out satisfaction (area) | 21.9 / 19.0 % | **29.0 / 29.3 %** | 20.8 / 24.3 % |

For comparison, the 13 human-chosen machine patches of the first A/B (arm B) gave:

- within 2 voxels: +3.5 points (seed 1) and +3.1 (seed 2);
- fitter's held-out satisfaction: 24.4 % (seed 1) and 28.1 % (seed 2).

Per held-out patch, seed 1, nodes within 2 voxels (A → S):

| patch | s001 | s002 | s010 | s012 | s015 | s026 | s043 |
|---|---|---|---|---|---|---|---|
| A → S | 365 → 398 | 181 → 187 | 561 → 526 | 217 → 216 | 483 → 580 | 125 → 113 | 130 → 162 |

S is better on four of the seven. s001, s015 and s043 are also the three patches the human-chosen patches helped most.

## Reading

- **No headline gain.** The self-patch loop does not make a large, sure gain on this band, and by the bar fixed in
  advance it is a NO-GO.
- **The effect is real in sign.** Three things point the same way:
  - S-A is positive on both seeds, about the size the human-chosen patches gave;
  - random tiles from the same fit do not help, and slightly hurt;
  - the gap S-R is +3.4 points on both seeds, with the seed-2 CI above 0.
- **What this suggests.** The useful part is the selection rule (seed agreement plus a single sheet), not re-feeding
  the fit to itself. The fitter's own held-out satisfaction agrees: S sits at 29 % on both seeds, against 19-22 % for
  tracks only and 21-24 % for random tiles.
- **Limits.**
  - One band of one scroll.
  - Seven held-out patches, all in one z group.
  - Two seeds.
  - The selector's thresholds were set once, by hand, from coverage numbers only, and were not tuned.

The self-patch and random tiles are PHerc0813 surface geometry and are not released. Their ids, node counts and z
ranges are in `results/selfpatch_select.json` and `results/selfpatch_random.json`. Score files:
`results/selfpatch_{S_vs_A,R_vs_A,S_vs_R}{,_s2}.json`.

One harness note: the band's tracks DBM and Lasagna mirror had been deleted from C: between 26 Oct and 9 Nov. They
were rebuilt with the same tools and arguments (`jobs/rebuild_inputs.sh`). The cut matches the 23 Oct one exactly:
1,270,348 tracks, 47,877,707 points and the same source-id SHA-1 `efe3a8f0…`. The normals pool is the same 9,762
bricks.
