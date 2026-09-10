# SpeXTool 5.0.3 behavioral-reference map

This document identifies the existing IDL routines to consult while reimplementing the supported
workflow. It is a navigation aid, not permission to copy source text.

## Instrument configuration and raw detector data

- `instruments/ishell/data/ishell.dat` — iSHELL defaults used by xspextool
- `instruments/ishell/pro/mc_readishellfits.pro` — iSHELL MEF read, variance/mask, detector corrections
- `instruments/ishell/pro/mc_ishellampcor.pro` — amplifier correction
- mode/reference assets under `instruments/ishell/data/`

## Image combination / darks

- `pro/mc_combimgs.pro` — image combination statistics
- supporting robust/statistical routines called by it

## Flat calibration

- `instruments/ishell/pro/mc_ishellcals2dxd.pro` — iSHELL calibration orchestration
- `pro/mc_normspecflat.pro` — flat normalization
- `pro/mc_findorders.pro` and flat-info readers — order location/reference information

## Wavelength calibration and rectification

- `instruments/ishell/pro/mc_ishellcals2dxd.pro` — orchestration
- `pro/mc_wavecal1dxd.pro` — simultaneous cross-dispersed 1-D wavelength solution
- `pro/mc_findlines1dxd.pro` — line identification support
- `pro/mc_findlines2d.pro` — 2-D line traces
- `pro/mc_fitlinecoeffs1d.pro`, `pro/mc_fitlinecoeffs2d.pro` — line-shape/distortion fits
- `pro/mc_mkrectindcs2d.pro` — rectification indices
- `pro/mc_mkwavecalimgs2d.pro` — wavelength/spatial calibration images
- `pro/mc_rectifyorder.pro`, `pro/mc_rectorder.pro` — order rectification

## Point-source extraction

- `pro/xspextool.pro` — workflow orchestration and defaults/state transitions
- `pro/mc_mkspatprof.pro` — spatial profiles
- `pro/mc_findpeaks.pro` — automatic aperture positions
- `pro/mc_tracespec.pro` — wavelength-dependent trace
- `pro/mc_mkapmask.pro` — aperture/background masks
- `pro/mc_extpsspec.pro` — point-source extraction
- supporting spatial-model/optimal-extraction routines called by `mc_extpsspec.pro`

For the automatic one-aperture path, `mc_findpeaks.pro` finds the maximum of the absolute,
median-subtracted spatial profile, fits a Gaussian around it, and records the fitted center/sign.

## Combining spectra

- `pro/xcombspec.pro` — combination workflow
- `pro/xmc_scalespec.pro` — selected-order scaling UI/orchestration
- `pro/mc_getspecscale.pro` — scale-factor calculation
- combination/statistical helpers used by xcombspec

## Telluric correction

- `pro/xtellcor.pro` — supported workflow
- `pro/mc_mktellspec.pro` — construction of telluric correction spectrum
- `pro/mc_vegacorr.pro`, `pro/mc_vegaconv.pro` — Vega model handling
- iSHELL `IP_coefficients.dat` and xtellcor mode information — IP path

v0.1 intentionally omits the optional residual wavelength-shift procedure.

## Order merging

- `pro/xmergeorders.pro` — merge workflow
- `pro/mc_mergespec.pro` — numerical merge of adjacent spectral segments
