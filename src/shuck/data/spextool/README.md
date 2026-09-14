# Bundled Spextool reference assets

- Upstream package: Spextool
- Upstream version: 5.0.3
- Authors: Michael C. Cushing, William D. Vacca, and John T. Rayner
- Imported: 2026-09-14
- Import source: the byte-for-byte local upstream tree at `reference/Spextool` in the shuck
  development workspace; `data/version.dat` identifies that tree as version 5.0.3
- Purpose: allow installed copies of shuck to run the supported iSHELL J3/Kgas workflow without an
  external Spextool installation

The following original upstream files are copied with their relative paths preserved:

- `data/version.dat` and `data/vega500000.sav`
- `instruments/ishell/data/ishell.dat` and `xtellcor_modeinfo.dat`, which record upstream iSHELL
  processing and telluric-mode configuration
- `instruments/ishell/data/ishell_bias.fits`, `ishell_lincorr_CDS.fits`,
  `ishell_bdpxmk.fits`, and `ishell_htpxmk.fits`, used for detector correction and masking
- `instruments/ishell/data/J3_flatinfo.fits` and `Kgas_flatinfo.fits`, used for order geometry and
  normalized-flat construction
- `instruments/ishell/data/J3_wavecalinfo.fits`, `Kgas_wavecalinfo.fits`, `J3_lines.dat`, and
  `Kgas_lines.dat`, used for ThAr wavelength calibration and rectification
- `instruments/ishell/data/IP_coefficients.dat`, used with the Vega model for telluric correction

`SHA256SUMS` records the upstream-file checksum of every copied asset. The imported files have not
been numerically modified.

These files are selected runtime data, not the complete Spextool distribution. In particular, the
external IDL libraries historically bundled under Spextool's `other/` directory are not included.
IDL source, manuals, atmospheric grids, reference-atlas PDFs, unsupported-mode calibrations,
editor backups, and operating-system metadata are also excluded because shuck does not load them.

Please cite Cushing, M. C., Vacca, W. D., & Rayner, J. T. 2004, "Spextool: A Spectral Extraction
Package for SpeX, a 0.8–5.5 Micron Cross-Dispersed Spectrograph," PASP, 116, 362,
[https://doi.org/10.1086/382907](https://doi.org/10.1086/382907).

See the repository's `THIRD_PARTY_NOTICES.md` for permission, licensing, telluric-paper citation,
and Vega-model provenance. This material is not represented as being relicensed under shuck's MIT
License.
