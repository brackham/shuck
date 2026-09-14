# `.shuck` control-file specification — draft format version 1

The `.shuck` file is the complete human-readable recipe for a reduction. `shuck setup` may infer
and propose values, but all science/calibration associations used by the reduction must be explicit
in the file.

The format is INI-like configuration sections plus pipe-delimited tables. Blank lines and lines
beginning with `#` are ignored.

## Standard night layout

Format version 1 uses this portable directory layout:

```text
<NIGHT>/
├── raw/
├── proc/
├── cal/
├── qa/
├── <NIGHT>.shuck
├── <NIGHT>.overrides.toml
└── <NIGHT>.obslog.csv       # only with --write-log
```

`raw/` must already exist and contain the input FITS files. `shuck setup` creates `cal/`, `proc/`,
and `qa/` when needed, but never creates `raw/`. Setup accepts either `<NIGHT>` or `<NIGHT>/raw` as
its input and, unless `--output` is supplied, writes `<NIGHT>/<NIGHT>.shuck`. It does not search
other subdirectories recursively.

Raw FITS files are immutable inputs: setup never modifies them. The `.shuck` file is a generated
reduction plan, `<NIGHT>.overrides.toml` stores persistent human classification decisions, and the
optional `<NIGHT>.obslog.csv` is derived header metadata. An observatory-provided `obslog.txt` is
left untouched and is not an authoritative setup input in format version 1.

## Setup overrides and observing log

On the first setup run, when `<NIGHT>.overrides.toml` is absent, setup writes the normally inferred
`.shuck` plan and a commented TOML file containing any conservative object-name suggestions. The
suggestions are not applied on that invocation. The user reviews, edits, or deletes the proposed
rules and reruns setup; the existing override file is then read before the final classifications are
built and is never modified by setup. A default-output rerun regenerates `<NIGHT>.shuck`; an
existing explicitly selected `--output` still requires `--overwrite`.

The v0.1 override schema uses exact object names and filenames:

```toml
[objects."HD_106965"]
frametype = "standard"
bmag = 7.63
vmag = 7.54
rv_kms = -8.36

[files."icm.2026A021.260406.TOI3884.00030.a.fits"]
object = "HD_106965"
frametype = "standard"
```

Object rules support `frametype = "science"`, `"standard"`, or `"ignore"`, because they apply to
target identities. Standard-star object rules may also record numeric `bmag`, `vmag`, and `rv_kms`
values; setup carries these explicitly into the standards table instead of emitting placeholders.
File rules support `object`, `frametype`, or both; a file rule may use any supported raw-exposure
role. A file-level object correction establishes the resolved object name; the matching object rule
is then considered, and a file-level `frametype` has final precedence:

```text
file frametype > resolved-object frametype > normal header/setup inference
```

The default file is discovered at `<NIGHT>/<NIGHT>.overrides.toml`; `--overrides PATH` selects an
explicit file. Overrides never alter the FITS headers. Setup reports which object or file rule
supplied a final value and reports unmatched rules for review.

Object-name heuristics only propose first-run corrections. An object beginning with `HD` suggests
`standard`; another ordinary target object suggests `science`. Names explicitly containing
standard/telluric labels are left for review rather than forced through the non-HD heuristic. A
disagreement or mixed science/standard
metadata produces an object-level review summary. These heuristics do not apply to flats, arcs,
darks, or other calibration frames. For filenames following the iSHELL structured naming pattern,
a filename/header object disagreement is reported, but the FITS `OBJECT` value is retained unless a
file override explicitly changes it. `OBJECT` and `TCS_OBJ` are treated as aliases when they agree
or only one is present for target exposures; conflicting values remain unresolved for review rather
than being silently chosen. For `DATATYPE=calibration`, `OBJECT` identifies the calibration role and
a different `TCS_OBJ` is normal iSHELL bookkeeping, not a review condition.

With `--write-log`, setup writes `<NIGHT>/<NIGHT>.obslog.csv` with one row for every FITS file
directly inside `raw/`. `filename` is the first column. Remaining columns are the deterministic
first-seen union of all scalar cards in all HDU headers; absent cards become empty CSV fields.
When present, `OBJECT`, `TCS_OBJ`, `DATATYPE`, `ITIME`, `XDTILT`, and `BEAM` follow `filename` in
that order; all other columns retain their deterministic first-seen order.
Primary-header keywords retain their simple names, while extension keywords use `HDU<N>.KEYWORD`
to avoid collisions. Repeated cards receive deterministic `#<N>` suffixes. Blank, `COMMENT`,
`HISTORY`, and `CONTINUE` cards are omitted. Rows use the setup observation order: `MJD_OBS` when
all files provide it, otherwise natural filename order. Standard CSV quoting is used.

## Required top-level configuration

```text
[shuck]
format_version = 1
raw_path = raw
calib_dir = cal
proc_dir = proc
qa_dir = qa
```

These paths are relative to the control file. The default control-file location therefore keeps the
entire night directory portable when it is moved as a unit. When an explicit output path is outside
the night directory, setup writes paths relative to that control-file location.

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

`shuck setup` always writes the `[telluric]` section. A format-version-1 parser may accept the
section as absent and substitute the fixed v0.1 values shown above.

## Tables

### `calibrations read` / `calibrations end`

One row per distinct, complete flat+wavecal association. Required fields:

`calib_id | mode | flat_files | arc_on_files | arc_off_files`

The three file fields are comma-delimited lists and each must contain one or more files. A
calibration set contains a same-mode sequence of flats and compatible ThAr lamp-on and lamp-off
exposures. Setup identifies separate sequences conservatively from observation/file order; it must
not merge every calibration exposure of a mode across an entire night. Incomplete or ambiguous
sequences remain unresolved for user review.

Setup orders the raw stream by `MJD_OBS` when every scanned FITS file provides it, otherwise by
natural filename order. A non-calibration exposure or a mode change ends the current candidate
sequence, and a new flat after arc exposures starts a new candidate. Only candidates containing at
least one flat, arc-on exposure, and arc-off exposure become calibration sets.

Setup assigns the uniquely nearest complete same-mode calibration set in observation time to a
science or standard exposure, without requiring a matching `TCS_OBJ` or pointing. A tie or other
ambiguity leaves the `calib` field blank. Every proposed assignment is written explicitly in the
data table.

### `darks read` / `darks end`

One row per master-dark group. Required fields:

`dark_id | itime | ndr | coadds | files`

Dark groups match exactly on `(ITIME, NDR, CO_ADDS)`; format version 1 does not apply an
integration-time tolerance. A group with fewer than five files produces a warning, but dark-frame
count alone is never fatal.

### `standards read` / `standards end`

One row per telluric standard identity. Required fields:

`standard_id | target | bmag | vmag | rv_kms`

### `data read` / `data end`

One row per relevant raw exposure. Required fields:

`filename | frametype | target | mode | beam | calib | dark | comb_id | telluric_group`

`frametype` explicitly records the raw exposure role. Supported values are `flat`, `arc_on`,
`arc_off`, `dark`, `science`, `standard`, and `ignore`. Calibration and dark frames also appear in
their dedicated grouping tables; their data rows retain the raw-frame role rather than using
`ignore`.

Setup classifies `DATATYPE=calibration` before applying target rules: `OBJECT=QTH` is `flat`,
`OBJECT=ThAr on` is `arc_on`, and `OBJECT=ThAr off` is `arc_off`. A dark is confidently
recognized when `DATATYPE=calibration`, its filename contains `dark`, and normalized `SLIT=Mirror`;
such frames are `dark`, and their normal `OBJECT=no_name` metadata does not create object-identity
review notes. Likewise, recognized `flat`/`QTH` and `arc`/`ThAr on` or `arc`/`ThAr off` filename
and `OBJECT` pairs are expected calibration metadata, not object-identity review conditions.
Only target science/standard B-beam rows default to `ignore`. Thus a normal B-beam ThAr-off frame
remains `arc_off`. `DATATYPE=target` versus `standard` remains reviewable target metadata and may
be corrected through overrides; it does not supersede calibration classification.

`comb_id` identifies exposures that will be spectrally combined. Exposures in one combination group
may reference different `dark` IDs when their exact detector/exposure configurations differ; there
is no one-to-one relationship between `comb_id` and `dark`.

For a science row, `telluric_group` is the `comb_id` of the standard-star exposures that are
combined to form its telluric standard spectrum. Standard and ignored rows leave
`telluric_group` blank.

## Validation principles

Fatal errors include missing files, unsupported modes, incomplete or incompatible calibration
assignments, science or standard A frames without darks, unresolved telluric associations, and
invalid table references. Warnings include dark groups with fewer than five exposures, suspicious
wavecal metrics, and large object-standard airmass differences.
