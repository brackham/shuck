# Numerical validation plan

## Principle

A visually reasonable final spectrum is not sufficient. Each major Python stage should be compared
against the corresponding SpeXTool 5.0.3 product for the same raw data and choices.

## Local reference-data layout

Full-resolution raw and SpeXTool products remain outside Git. Point to them with:

```bash
export SHUCK_REFERENCE_DATA=/path/to/shuck-reference/260406
export SPEXTOOL5_DIR=/path/to/Spextool
```

Suggested private layout:

```text
$SHUCK_REFERENCE_DATA/
  raw/
  spextool_cal/
  spextool_proc/
  manifests/
```

Regression tests should skip with an explicit message when the environment variable is unavailable.

## Comparison sequence

1. Raw MEF reading and detector-corrected image/variance/mask.
2. Master darks for representative exposure configurations.
3. J3 and Kgas normalized flats.
4. J3 and Kgas 1DXD wavelength solutions and 2-D rectification products.
5. One representative J3 extracted science exposure.
6. One representative Kgas extracted science exposure.
7. Representative extracted A0V exposures.
8. J3 and Kgas combined science/standard spectra.
9. J3 and Kgas telluric correction spectra and corrected science spectra.
10. J3 and Kgas merged spectra.

For every regression, report shape compatibility, finite/masked-pixel agreement, absolute and relative
residual statistics, and scientifically relevant fit metrics. Do not choose final tolerances until the
first independent Python implementation has been compared and discrepancies understood.

## Known reference caveat

Do not make a regression test reproduce known manual bookkeeping errors in the existing Kgas A0V
reduction. The target is SpeXTool's numerical algorithm applied to the correct raw-frame/dark
associations, not an accidental duplicate or incorrect association.
