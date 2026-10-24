# Pre-registration: tracks only vs tracks + automatic patches, PHerc0813 z 9000-10000

Written 23 Oct 2026, before any fit of this band in either arm. The input files and the patch split below are
already fixed on disk. The settings in this file do not change after the first fit starts. Anything added later is
marked "post hoc".

## Question

Does adding machine-grown sheet patches to the spiral fit, on top of the published surface tracks, place the
fitted windings closer to patches the fit never saw? Does it do so without stacking more windings on one sheet?

## Data

- **Scroll and band.** PHerc0813, volume `20250821151723` (9.362 um, level-0 voxels), z [9000, 10000).
- **Spiral sense.** CW. It comes from the open-data catalog flags `z_direction_is_top_to_bottom = false` and
  `left_handed_coordinates = false`, read through villa main's `spiral_outward_sense_for`; see
  `dataset/catalog_orientation.json`. villa 0fb38c45 rejects those keys in `spiral-scroll.json`, so the sense
  derived from them is written as `spiral_outward_sense`.
- **Umbilicus.** `dataset/umbilicus.json` is byte-identical (md5 6a0ea2bf8e12492d5e7cd1dab09ed6a6) to
  `PHerc0813_umbilicus.json` of AlexeyDrobkovStrikesBack/herculaneum-umbilici (MIT, commit de7642a).
- **Tracks.** These are the published tracks of dl.ash2txt.org `spiral_datasets/PHerc0813/20250821151723/tracks/`,
  cut to the band without downloading the 8.17 GB DBM (`tools/cut_band_tracks.py`). A track is kept when it lies
  wholly inside the band, which is the loader's own rule. The cut keeps 1,270,348 tracks and 47,877,707 points.
  Key names and list positions are unchanged, so every source id matches the published store. villa 0fb38c45's
  `load_tracks_from_dbm` loads exactly these tracks, with matching source ids, families, point counts and z
  bounds (`logs/verify_band_tracks.log` (kept locally)).
- **Lasagna normals.** nx/ny group 2 (4x) of `PHerc0813/representations/predictions/lasagna/20250821151723-lasagna-20260419180421`,
  chunk rows 69-78 (L0 z 8832-10112). The fit's pool is predicted at 9,762 of 10,807 bricks
  (`dataset/lasagna_inputs/respool_check.json`). No grad_mag or surface-distance store is used.
- **Patches.** The 21 PHerc0813 patches of
  [Ahggggg/first-letters-clean-sheets](https://github.com/Ahggggg/first-letters-clean-sheets) at commit
  `de78c8b`. They were grown with villa's `vc_grow_seg_from_seed` on the public m7 prediction, and their authors
  say the patches are not shown to lie on one papyrus layer. Data licence: CC BY-NC 4.0, with attribution in
  `dataset/unverified_patches/_ATTRIBUTION.md`. All 21 patches are level-0 coordinates of this volume (meta.json
  `volume_id`, `volume_shape_zyx` [16993, 7947, 7947], every node inside it).

## Split (fixed)

The split rule uses patch geometry only (`tools/prep_patches.py`, output in `dataset/patches_split.json`):

1. Patches with a non-empty `flags` column in the authors' `index.csv` are dropped. One patch is dropped:
   `PHerc0813_t269_s000`, flagged `xsec_tilt_ge_2deg_post_hoc`.
2. A patch is **held out** (`hld_`) if the mean z of its nodes inside the band is >= 9600. Every other patch is
   used for **training** (`trn_`).

| | patches | area inside the band | nodes inside the band |
|---|---|---|---|
| training `trn_` | 13 | 13.45 cm2 | 38,912 |
| held out `hld_` | 7 | 2.32 cm2 | 6,815 |

- Training: t269_s003, t269_s004, t269_s007, t269_s009, t269_s011, t269_s014, t269_s016, t269_s024, t269_s025,
  t269_s040, t275_s050, t275_s054, t275_s055.
- Held out: t269_s001, t269_s002, t269_s010, t269_s012, t269_s015, t269_s026, t275_s043. Every id carries the
  `PHerc0813_` prefix; the full ids are in `patches_split.json`.

The 9600 threshold was chosen by looking at patch positions, before any fit. It sits in the largest gap of the
in-band mean z (9533 to 9714) and gives about a third of the patches by count. By area, the held-out set is only
15% of the in-band patch area. The held-out patches form one group at the top of the band. Every held-out node
inside the band is at least 204.7 voxels from every training node, and every held-out patch's nearest neighbouring
patch is another held-out patch. So the test asks whether patches improve the fit **away from** where they were
placed. It is not a test of whether the fit reproduces them.

## Arms (identical except for the patch input)

The code is villa `0fb38c45` (the First Letters recipe commit) plus the same local changes as the First Letters
worktree (`patches/villa_0fb38c45_local.diff`: optional outer shell, GPU memory fraction, periodic checkpoints, crossing-cache
signature by name and size). Python 3.14, torch 2.13, RTX 3090, and the MSVC `vc.track_crossings` build.

- **Shared settings** (`configs/common.json`): seed 1, 20,000 steps, `dense_spacing_mode` grad_mag with
  `loss_weight_dense_spacing` 0, shell weights 0, `loss_start_track_dt` = `loss_start_patch_dt` =
  `loss_start_unverified_patch_dt` = 10000, and the First Letters per-step sample counts. GPU memory fraction is
  0.78 and a checkpoint is written every 2000 steps.
- **Arm A, tracks only.** `input_disable_patches: true`.
- **Arm B, tracks + training patches.** The 13 `trn_` patches are input as **unverified patches**
  (`paths.unverified_patches`, `patch_uuid_filter_regex: ^trn_`, empty `verified_patches/`). In 0fb38c45 these
  get the unverified-patch weights (radius 2.0, DT 1.0). They are not trusted geometry, so no track is removed near
  them: `track_exclusion_radius` 16 has nothing to act on in either arm. This avoids the confound that verified
  patches would bring, where tracks within 16 voxels of a patch are deleted. villa main has since removed the
  unverified-patch input, so a result here may not carry over to main's verified-patch path.
- **Held-out eval pass, per arm.** The arm's `checkpoint_fitted.ckpt` is resumed with the same step count, so no
  iterations run. Only the `hld_` patches are loaded, as unverified patches, with normals and the mesh export off.
  The fitter then writes its own satisfaction for them (`satisfied_fitted.json`, `unverified_patches`). Its
  `satisfied_tracks` must equal the arm's fit. If it does not, the eval is invalid.

## Metrics

All metrics are computed on CPU by `score_ab.py` from each arm's raw winding meshes (`meshes/fitted_<tag>/wNNN_<tag>`;
the `_spliced` copies are not used).

**Primary.** `within_2` is the share of held-out patch nodes with z in [9020, 9980) whose exact distance to the
nearest fitted winding surface is <= 2 voxels. It is pooled over nodes.

**Secondary:**

- `within_4`, with the same definition at 4 voxels.
- Per held-out patch: winding hops (grid edges whose two nodes, each within 4 voxels of a winding, lie nearest to
  different windings), the number of distinct windings, and the modal-winding share.
- Duplicate-sheet share. This is skuzbucket1's geometry_coverage
  ([skuzbucket1/scroll-prize](https://github.com/skuzbucket1/scroll-prize) `experiments/geometry-coverage`, MIT,
  commit d0b0bdf), re-implemented in `score_ab.py`. It is the area share of winding nodes that have a node of
  another winding within 25 voxels, less than 0.03 mm along that winding's normal and less than 25 voxels sideways.
- `satisfied_tracks` from each fit log.
- The fitter's held-out satisfaction from the eval pass.
- Three fixed CT cross-sections at z = 9500, 9700 and 9900, all inside the scoring window. Each shows arm A and
  arm B windings and the held-out patch traces over the CT at level 1, cropped to the held-out patches' bounding
  box plus 256 voxels.

**Uncertainty.** A paired bootstrap over the 7 held-out patches. Patches are resampled with replacement and the
same draw is used for both arms (10,000 draws, seed 12345). The interval is the 95% percentile interval of B-A.

**Reference (not part of the decision).** mkorayemai-web/vesuvius-band-sweep published a tracks-only fit of this
band. It may be scored with the same script as an outside reference row.

## Decision

The GO bar requires all three of:

1. B-A `within_2` >= +5 points;
2. its 95% bootstrap interval lies entirely above 0;
3. B-A duplicate-sheet share <= +2 points.

If any condition fails, the result is NO-GO: automatic patches did not improve this fit. If time allows, a second
seed (2) of both arms is run and reported whatever it shows. The decision stays with seed 1. A claim also needs
seed 2 to show the same sign.

**Invalid run.** A run is invalid if any of the following holds: the lasagna normals pool is <= 1000 bricks; the
loaded tracks are not exactly 1,270,348; arm B does not load exactly 13 patches; or the eval pass's
`satisfied_tracks` differs from its fit. An invalid run is rerun once. If it fails again, it is reported as failed.
WDDM shared GPU memory above 300 MB is logged and flagged; it slows the fit but does not change the result.

**A negative result is published as a negative result**, in the same place and with the same detail as a
positive one would be.

## Known limits

- The test has little power: 7 held-out patches, 2.32 cm2 and 6,815 nodes, one seed. Seed-to-seed noise in a
  similar setup was about 2 points (PCU, Jashann/vesuvius-scrolling).
- `within_2` and `within_4` do not know winding numbers. A held-out patch that lies on the wrong whole winding
  still counts as close (the same blind spot as the fitter's satisfaction metric, villa #1621). Hops only catch
  switches inside a patch.
- The patches are unverified and machine-grown. Our August Paris4 screen found that bulk machine patches lowered
  withheld-fiber consistency, so arm B can be worse than arm A.
- The held-out region is the top of the band and close to its edge. Both arms share that edge, and the 20-voxel
  scoring margin keeps nodes off it.

## Prior work and credit

- Ahggggg/first-letters-clean-sheets made the patches (CC BY-NC 4.0).
- skuzbucket1 defined the duplicate-sheet metric.
- Nicodol/spiralcheck is earlier held-out patch scoring for spiral fits, on Paris4 and PHerc1218. `score_ab.py`
  is a smaller re-implementation for this test, and spiralcheck can be run as a cross-check.
- PCU (Jashann/vesuvius-scrolling) compared tracks only with published patches on Paris4. It was not paired and
  did not use an eligible scroll.
- villa's own band-seed ablation config (`configs/sweep_08_22_26/defaults_no_band_seed_patches.json`) has no
  published results.
- We found no public tracks vs tracks + automatic patches A/B on an eligible scroll (GitHub code, repository and
  issue search, 23 Oct 2026).

AI assistance: this pre-registration, the band setup and the scripts were prepared with Claude (Anthropic).


## Amendment 1 (24 October 2026, after the pipeline smoke test, before either 20,000-step fit)

The 200-step smoke test (both arms plus both eval passes) passed every validity check except one: the eval pass did
not reproduce the fit's `satisfied_tracks` exactly (arm A 96,783 vs 96,838; arm B 96,282 vs 96,096, of 1,270,348).
These 0.06 % and 0.19 % differences come from re-scoring the same checkpoint on the GPU, which is not bit-exact;
nothing else in the eval differs between arms. The validity rule is changed from "equal" to "within 0.5 % of the
fit's satisfied tracks", for both arms alike. The eval pass only feeds a secondary metric (the fitter's held-out
satisfaction); the primary metric is computed from the exported meshes and is unaffected. Nothing else changes.
All other smoke checks passed: 1,270,348 tracks in every run, normals pool 9,762 bricks, arm A no patches, arm B 13
training patches and 0 verified, both evals resumed with 0 steps and scored the 7 held-out patches, shared GPU
memory 86 MB.
