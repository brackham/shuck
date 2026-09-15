# shuck

`shuck` is a scriptable, non-interactive Python reduction pipeline for NASA IRTF/iSHELL spectra.
Its current focus is reproducing the scientifically relevant behavior of Spextool 5.0.3 for a narrow 
workflow with inspectable intermediate products.

Shuck is an independent project. It is not an official NASA IRTF or Spextool product.

## Installation

`shuck` is not yet available on PyPI. For now, install it directly from GitHub.

Because the bundled Spextool calibration/reference assets are managed with Git LFS, install and
initialize Git LFS first:

```bash
git lfs install
pip install "git+https://github.com/brackham/shuck.git"
```

For development from a source checkout:

```bash
git clone https://github.com/brackham/shuck.git
cd shuck
git lfs pull

python -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

## Quick start

The standard night layout has an existing `raw/` directory. Setup creates sibling `cal/`, `proc/`,
and `qa/` directories as needed.

```bash
shuck setup /path/to/night --write-log
# Review/edit /path/to/night/night.overrides.toml.
shuck setup /path/to/night
shuckit /path/to/night/night.shuck
```

Passing `/path/to/night/raw` to `shuck setup` is equivalent. Use `-o FILE.shuck` to override the
default control-file path. Developer/power-user stage commands are also available:

```bash
shuck calibrate night.shuck
shuck extract night.shuck
shuck combine night.shuck
shuck telluric night.shuck
shuck merge night.shuck
```

See [the control-file specification](docs/CONTROL_FILE_SPEC.md) for the complete format and
association rules.

## Current scope

Version 0.1 supports iSHELL J3 and Kgas point-source observations reduced as A-Sky/Dark, with one
positive aperture, optimal extraction, ThAr wavelength calibration and rectification, spectral
combination, A0V/Vega telluric correction using the iSHELL instrument profile, and order merging.

Deliberate exclusions include other iSHELL modes, A-B and extended-source extraction, multiple
apertures, spectral shifting/pruning/shape correction, residual telluric shift fitting, xcleanspec,
and GUI interaction. The pipeline has unit and synthetic numerical coverage and has run end to end
on the private 260406 night; direct Spextool regression work remains part of science validation.

## Relationship to Spextool

Shuck's algorithms and bundled calibration/reference assets derive from, and are intended to
reproduce the relevant behavior of, Spextool 5.0.3. Selected upstream assets are bundled with
permission so installed copies of shuck are self-contained for the supported workflow. Original
filenames and layout are retained where practical, and a checksum manifest makes the import
auditable.

## Citation

Users of `shuck` should cite:

Cushing, M. C., Vacca, W. D., & Rayner, J. T. 2004,
"Spextool: A Spectral Extraction Package for SpeX, a 0.8–5.5 Micron Cross-Dispersed
Spectrograph," PASP, 116, 362.
[https://doi.org/10.1086/382907](https://doi.org/10.1086/382907)

Because `shuck` implements the Spextool telluric-correction procedure, users of that procedure should
also cite:

Vacca, W. D., Cushing, M. C., & Rayner, J. T. 2003,
"A Method of Correcting Near-Infrared Spectra for Telluric Absorption," PASP, 115, 389.
[https://doi.org/10.1086/346193](https://doi.org/10.1086/346193)

The Spextool documentation requests citation of both papers when its telluric algorithms are used.

## License and bundled material

Shuck's original code is available under the [MIT License](LICENSE). Bundled Spextool-derived
material is not represented as MIT-licensed; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
and the provenance README under `src/shuck/data/spextool/` for its attribution and terms.
