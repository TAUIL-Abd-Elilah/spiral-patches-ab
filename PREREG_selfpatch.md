# Pre-registration: the spiral fit's own "good" regions as patch inputs to a refit (PHerc0813 z 9000-10000)

Written 9 Nov 2026, before any fit of arms S or R. The patch sets below are fixed on disk and listed in
`results/selfpatch_select.json` and `results/selfpatch_random.json`. The settings here do not change after the first S
or R fit starts. Anything added later is marked "post hoc".

## Question

The team's winding notes list as not yet tried: *identifying methods to automatically crop "good" regions of the spiral
fit, and using these as surface patch inputs to a subsequent run.* This test asks one narrow version of that question.
Take a tracks-only fit and keep only the regions that two independent seeds agree on, with no second winding stacked
on the sheet. Feed those regions back as patches and refit. Do the fitted windings then land closer to human-traced
patches that no fit and no selector ever saw? And is the selection what matters, or would any re-fed tiles do?

## Harness (unchanged from PREREG.md)

Everything is the harness of `PREREG.md` (the patches A/B of 23 Oct): PHerc0813, volume `20250821151723`,
z [9000, 10000), the 1,270,348 published tracks, the Lasagna normals pool, villa `0fb38c45` with the same local diff,
20,000 steps, `configs/common.json`, GPU fraction 0.78, checkpoint every 2,000 steps, and the eval pass on the
held-out patches (`configs/eval_hld.json`). The 7 held-out patches (`hld_`, Ahggggg/first-letters-clean-sheets at
`de78c8b`, CC BY-NC 4.0) are the truth. Arm A is the existing tracks-only fit of 24 Oct (seed 1) and 26 Oct (seed 2).
It is not re-run.

## Arms

- **A**: tracks only (existing runs `patchA`, `patchA_s2`).
- **S**: tracks plus the self-patches `slf_*` as unverified patches (`configs/arm_S.json`, filter `^slf_`).
- **R** (control): tracks plus the same number of random tiles `rnd_*`, cut from the same fit with the same tile grid
  but ignoring the selector (`configs/arm_R.json`, filter `^rnd_`).

S and R are each fitted with seed 1 and seed 2, everything else identical. The patch sets are the same for both
seeds.

## Selector (`tools/self_patches.py`; no human patch is read)

Inputs: the winding meshes of arm A seed 1 (`fitted_patchA`) and arm A seed 2 (`fitted_patchA_s2`). A node of a seed-1
winding is **good** when all of these hold:

1. **Seed agreement.** The nearest seed-2 winding surface is within 2 voxels (exact point-to-triangle distance) and
   has the same winding index.
2. **Single sheet.** No node of another seed-1 winding is within 25 voxels sideways and closer than 0.03 mm along the
   normal. This is the duplicate-sheet rule of `score_ab.py`.
3. **In band.** The node is valid and inside z [9000, 10000).

**Tiles.** Each winding grid (rows along z, columns along the winding) is cut into blocks of 60 columns, full height.

- A block is kept when at least 60 % of its valid nodes are good.
- Nodes that are not good are set to -1, and only the largest 4-connected good component is kept.
- A tile needs at least 800 nodes.

**Random control.** The same number of tiles is drawn (numpy seed 0) from all blocks with at least 800 valid nodes, and
each is kept whole.

**The fixed sets** (from `results/selfpatch_select.json` and `results/selfpatch_random.json`):

| set | patches | windings | nodes | other |
|---|---|---|---|---|
| S (`slf_`) | 46 | 35 | 77,417 (~27 cm2 nominal) | 36.1 % of the 1,936,213 valid seed-1 nodes are good; 46.5 % agree with seed 2 on the same winding |
| R (`rnd_`) | 46 | 42 | 130,414 (~46 cm2 nominal) | drawn from 668 candidate blocks; R tiles are kept whole, so they are larger than S tiles |

All 92 tiles load with villa's own `tifxyz.load_tifxyz`. Every tile spans the full band height, so S and R tiles
also cover the held-out z range (z >= 9600). This is allowed, because the selector never reads a human patch. It is
also why R is the control that matters.

The selector's parameters were set by hand before any S or R fit and before any score. The only things looked at were
the selector's own coverage numbers (share of good nodes, number of tiles), never a held-out patch.

## Metrics and decision

Scoring is `score_ab.py`, unchanged, with the same settings as the A/B: `--prefix hld_`, scoring window
z 9020-9980, a paired bootstrap over the 7 held-out patches (10,000 draws, seed 12345).

- **Primary.** S-A, the share of held-out patch nodes within 2 voxels of a fitted winding, on seed 1.
- **GO** requires all three:
  - S-A within 2 voxels is at least **+3.0 points**, with its 95 % CI above 0, on seed 1;
  - the duplicate-sheet share S-A is at most +2 points;
  - the seed-2 S-A point estimate is above 0.

  +3 points is about what 13 human-chosen machine patches gave in the A/B (+3.5, and +3.1 on seed 2).
- **Selection matters** (secondary, reported but not part of GO): S-R within 2 voxels on seed 1, and the same on
  seed 2.
- **Also reported:**
  - within 4 voxels;
  - winding hops inside the held-out patches;
  - satisfied tracks;
  - the fitter's own held-out satisfaction from the eval pass;
  - S against B (the human-chosen patches) as context.

**Validity** is per run, as in the A/B job checks:

- rc 0 for the fit and its eval pass;
- 1,270,348 tracks loaded and a normals pool above 1,000 bricks;
- the right number of `slf_`/`rnd_` patches loaded, and 0 verified;
- the eval pass resumed at 20,000 steps, ran 0 steps and scored 7 held-out patches;
- eval satisfied tracks within 0.5 % of the fit;
- shared GPU memory at or below 300 MB.

An invalid run is re-run once with the same settings.

**Run order:** S seed 1, R seed 1, S seed 2, R seed 2.

## What will and will not be released

Code, configs, this file, the selector summaries (tile counts, ids, node counts, z ranges) and the scores. The
self-patch and random tiles are 0813 surface geometry and stay local, like the fitted meshes of the A/B.
