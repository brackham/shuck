# `.shuck` control-file specification — draft format version 1

The `.shuck` file is the complete human-readable recipe for a reduction. `shuck setup` may infer
and propose values, but all science/calibration associations used by the reduction must be explicit
in the file.

The format is INI-like configuration sections plus pipe-delimited tables. Blank lines and lines
beginning with `#` are ignored.

## Required top-level configuration

```text
[shuck]
format_version = 1
raw_path = raw
cal_path = cal
proc_path = proc
qa_path = qa
```

## Reduction defaults

```text
[extraction]
reduction_mode = A-Sky/Dark
n_apertures = 1
optimal = true
psf_radius = 1.0
aperture_radius = 1.0
background_start = 1.1
background_width = 2.0
background_degree = 0
trace_degree = 2

[combine]
statistic = robust_weighted_mean
sigma_clip = 8.0
J3_scale_order = 402
Kgas_scale_order = 224
shift_spectra = false
prune = false
shape_correction = false

[telluric]
method = IP
find_shifts = false
```

## Tables

### `calibrations read` / `calibrations end`

One row per flat+wavecal association. Required fields:

`calib_id | mode | flat_files | arc_on | arc_off`

### `darks read` / `darks end`

One row per master-dark group. Required fields:

`dark_id | itime | ndr | coadds | files`

### `standards read` / `standards end`

One row per telluric standard identity. Required fields:

`standard_id | target | bmag | vmag | rv_kms`

### `data read` / `data end`

One row per relevant raw exposure. Required fields:

`filename | frametype | target | mode | beam | calib | dark | comb_id | telluric_group`

`frametype` is one of `science`, `standard`, or `ignore` for v0.1. Calibration and dark frames are
represented in their dedicated tables rather than duplicated here.

## Validation principles

Fatal errors include missing files, unsupported modes, incompatible calibration assignments, science
or standard A frames without darks, unresolved telluric associations, and invalid table references.
Warnings include unusually small dark groups, suspicious wavecal metrics, and large object-standard
airmass differences.
