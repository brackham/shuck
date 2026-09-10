# Pipeline specification — v0.1

## Supported observations

- Instrument: iSHELL
- Modes: J3 and Kgas only
- Slit: read from FITS header; v0.1 is validated first on the 0.75 arcsec reference data
- Source: point source
- Beam strategy: A-Sky/Dark
- Apertures: one positive aperture
- Wavelength calibration: ThAr

## Routine workflow

1. `shuck setup RAW_DIRECTORY -o FILE.shuck`
2. User inspects/edits `FILE.shuck`.
3. `shuckit FILE.shuck` performs automatic validation and all stages below.

Stage commands remain available independently for development and inspection.

## Calibration and reduction stages

### Detector read/correction

Reference behavior is SpeXTool 5.0.3 iSHELL FITS reading/correction. The implementation must
construct the science image, uncertainty/variance, and mask from the iSHELL MEF, including the
same amplifier, linearity, and saturation behavior used by the supported SpeXTool workflow.

### Master dark

Group by detector/exposure configuration, not by an assumed number of frames. Combine using the
SpeXTool default robust weighted mean with 8-sigma clipping. A small number of members is a warning,
not automatically an error.

### Normalized flat

Reproduce the supported SpeXTool flat path: detector-read/correction, order scaling, combination,
order-edge location, and 2-D order normalization. Flat products must retain the information needed
for subsequent order location/extraction.

### Wavelength calibration

For J3 and Kgas use ThAr on/off exposures and the mode-specific reference information. Reproduce
1DXD wavelength fitting, 2-D line distortion mapping, and rectification coordinates. Automatically
accept the normal cross-correlation offset; emit QA and warn on suspicious offsets/residuals.

### Science/standard preprocessing

For every A exposure: apply detector corrections, subtract the explicitly associated master dark,
divide by the explicitly associated normalized flat, and rectify with the associated wavecal.
B exposures are ignored in v0.1.

### Point-source extraction

- number of apertures: 1
- aperture finding: automatic
- tracing degree: 2
- optimal extraction: on
- PSF radius: 1.0 arcsec
- aperture radius: 1.0 arcsec
- background subtraction: on
- background start: 1.1 arcsec
- background width: 2.0 arcsec
- background polynomial degree: 0

Do not add adaptive aperture optimization until the reference implementation passes regression tests.

### Spectral combination

- statistic: robust weighted mean
- clipping threshold: 8 sigma
- scale to median spectrum
- determine one scale factor from order 402 (J3) or 224 (Kgas), then apply it to all orders
- no spectral shifting
- no pruning
- no spectral-shape correction

### Telluric correction

- A0V standard + Vega model
- iSHELL precomputed IP kernel path
- B and V magnitudes and standard radial velocity are explicit control-file metadata
- do not run the optional residual `Find Shifts` step

### Merge orders

Reproduce the normal xmergeorders behavior used in the reference reduction, with no manually selected
order override.

## Deliberately unsupported in v0.1

Other iSHELL modes, A-B source extraction, extended sources, multiple apertures, thermal-mode sky
wavecals, adaptive extraction parameters, xcombspec shifting/pruning/shape correction, xtellcor
residual shift fitting, xcleanspec, and GUI interaction.
