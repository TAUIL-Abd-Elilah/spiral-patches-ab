# spiral-patches-ab

**Do automatic sheet patches make the spiral fit better on a First Letters scroll?** A pre-registered, paired A/B of
villa's spiral fitter on PHerc. 0813 z 9000-10000: arm A uses the published surface tracks only; arm B adds 13
machine-grown single-sheet patches. Both arms are scored on 7 other patches that neither fit saw.

**Status (24 October 2026): pre-registered, fits queued, no results yet.** [PREREG.md](PREREG.md) fixes the data,
the patch split, the settings, the metrics and the decision rule before the first fit. Results, negative or
positive, will be added here with the same detail.

## Why

The spiral fitter places windings from tracks, patches, fibers and winding annotations, but on the scrolls still in
the First Letters prize only the tracks exist: verified patches, fibers and winding annotations are published for
PHerc. Paris 4 only. Fits on eligible scrolls hop windings or lay two windings on one sheet (about a fifth of the
fitted area duplicates another winding on PHerc. 0191 and 0813 in skuzbucket1's measurements). The spiral tutorial
says patches or fibers give "more bang for your buck" than winding annotations against sheet switches, and on
Paris 4 the maintainers already feed tens of thousands of automatically grown patches through the patch input. Nobody
had published whether automatic patches help a fit on an eligible scroll. This repository asks that, on one band,
with a held-out test.

## Design (details in PREREG.md)

| | |
|---|---|
| Scroll, band | PHerc. 0813, volume 20250821151723 (9.362 um), z [9000, 10000), spiral sense CW from the catalog flags |
| Code | villa 0fb38c45 (the recipe the First Letters fits use) + `patches/villa_0fb38c45_local.diff` |
| Tracks | the published tracks cut to the band by byte-range reads (`tools/cut_band_tracks.py`, 0.86 GiB fetched instead of the 8.17 GB store); the fitter loads exactly 1,270,348 tracks, as from the full store |
| Normals | published Lasagna nx/ny, group 2 |
| Patches | the 21 PHerc. 0813 patches of [Ahggggg/first-letters-clean-sheets](https://github.com/Ahggggg/first-letters-clean-sheets) (CC BY-NC 4.0; not redistributed here, ids and split in `dataset/patches_split.json`) |
| Split | 1 flagged patch dropped; held out if the mean z of its in-band nodes >= 9600: 13 training (13.45 cm2), 7 held out (2.32 cm2), every held-out node >= 204.7 voxels from every training node |
| Arms | identical (seed 1, 20,000 steps, DT losses from step 10,000) except arm B loads the 13 training patches as unverified patches |
| Primary metric | share of held-out patch nodes within 2 voxels of the nearest fitted winding |
| Decision | GO only if B-A >= +5 points, the paired bootstrap 95 % interval (over patches) is above 0, and the duplicate-sheet share rises by at most 2 points |

The test has little power (7 held-out patches, one seed), and the patches are machine-grown and unverified, so
arm B can be worse than arm A. Both outcomes are reported.

## Files

- `PREREG.md`: the pre-registration.
- `score_ab.py`, `tests/`: CPU scoring (exact point-to-triangle distances, winding hops, skuzbucket1's duplicate-sheet
  share re-implemented with credit, paired bootstrap, the GO rule, CT cross-sections); `python -m pytest tests`.
- `tools/cut_band_tracks.py`, `tools/verify_band_tracks.py`: cut a published track store to a z band without
  downloading it, and check the fitter loads exactly the same tracks.
- `tools/prep_patches.py`: the patch split. `tools/mirror_zarr*.py`, `tools/pack_lasagna.py`: the normals.
- `configs/`, `jobs/`: the fit settings and the GPU jobs (paths are this machine's; the fits run on one RTX 3090).
- `dataset/`: the scroll spec, catalog orientation, umbilicus (AlexeyDrobkovStrikesBack/herculaneum-umbilici, MIT)
  and the split.

## Credits

Patches: Ahggggg/first-letters-clean-sheets (CC BY-NC 4.0). Duplicate-sheet metric: skuzbucket1/scroll-prize
(geometry-coverage). Umbilicus: AlexeyDrobkovStrikesBack/herculaneum-umbilici. Held-out patch scoring for spiral
fits was done before on Paris 4 and PHerc. 1218 by Nicodol/spiralcheck; PCU (Jashann/vesuvius-scrolling) compared
tracks only with published patches on Paris 4. The fitter, tracks and normals are the Vesuvius Challenge team's
(ScrollPrize/villa).

Code MIT. Developed with AI assistance under my direction.
