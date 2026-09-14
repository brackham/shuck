# Third-party notices

The MIT License in `LICENSE` applies to shuck's original code and documentation. It does not
relicense the Spextool-derived material described below.

## Spextool 5.0.3

Shuck includes selected calibration and reference assets from Spextool 5.0.3 so that its supported
iSHELL J3/Kgas workflow can run without a separate Spextool installation. Spextool was written by
Michael C. Cushing, William D. Vacca, and John T. Rayner.

Spextool 5.0.3 was distributed without a formal software license. Michael Cushing has given the
shuck project permission to reuse and redistribute the relevant Spextool material, with the request
that users cite the Spextool paper. This permission is recorded here conservatively as a paraphrase;
it is not a claim that the upstream material has been relicensed under the MIT License.

The copied files retain their original filenames and relative directory structure under
`src/shuck/data/spextool/`. Their purpose, upstream paths, and checksums are documented in the
README and `SHA256SUMS` in that directory.

Users should cite:

- Cushing, M. C., Vacca, W. D., & Rayner, J. T. 2004, "Spextool: A Spectral Extraction Package
  for SpeX, a 0.8–5.5 Micron Cross-Dispersed Spectrograph," PASP, 116, 362.
  [https://doi.org/10.1086/382907](https://doi.org/10.1086/382907)

When using shuck's telluric-correction implementation, the Spextool documentation also requests
citation of:

- Vacca, W. D., Cushing, M. C., & Rayner, J. T. 2003, "A Method of Correcting Near-Infrared
  Spectra for Telluric Absorption," PASP, 115, 389.
  [https://doi.org/10.1086/346193](https://doi.org/10.1086/346193)

## Vega model provenance

The bundled `data/vega500000.sav` is the high-resolution Vega model distributed with Spextool and
used by its telluric-correction procedure. Vacca, Cushing, & Rayner (2003) describe the underlying
model as a Kurucz Vega spectrum at resolving power 500,000, scaled to the observed Vega flux at
5556 Angstrom reported by Megessier (1995). No separate license or notice is embedded in the IDL
save file. It is therefore retained as non-MIT Spextool material under the redistribution permission
described above, with this scientific provenance preserved.

## Material not copied

The upstream Spextool tree also contains convenience copies of external IDL libraries, including
the IDL Astronomy User's Library, Coyote Graphics routines, and Craig Markwardt's MPFIT/CM library.
Shuck does not use those libraries, and they are not included in this repository or its packages.
Spextool IDL source, manuals, atmospheric-model grids, reference atlases, and calibration files for
unsupported instrument modes are likewise not runtime dependencies and are not bundled.

Files carrying a distinct notice or provenance remain subject to that notice or provenance. Nothing
in shuck's MIT License supersedes third-party rights in the bundled scientific data.
