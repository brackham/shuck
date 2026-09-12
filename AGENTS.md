# AGENTS.md — shuck

## Project purpose

`shuck` is a Python reduction pipeline for NASA IRTF/iSHELL spectra. v0.1 is intentionally narrow:
J3 and Kgas modes, point sources, one aperture, A-Sky/Dark reduction, optimal extraction, ThAr
wavelength calibration, A0V/IP telluric correction, and order merging.

The goal of the port is numerical/scientific fidelity to the user's SpeXTool 5.0.3 workflow, not a
line-by-line translation of IDL and not a general replacement for all SpeXTool functionality.

## Sources of truth

Read these before changing scientific behavior:

1. `docs/PIPELINE_SPEC.md` — supported workflow and defaults.
2. `docs/CONTROL_FILE_SPEC.md` — `.shuck` grammar and associations.
3. `docs/PORTING_MAP.md` — SpeXTool routines defining reference behavior.
4. `docs/VALIDATION_PLAN.md` — numerical comparison requirements.

If code and docs disagree, stop and flag the discrepancy rather than silently changing the science.

## Scope rules

- Do not add other iSHELL modes unless explicitly requested.
- Do not add A-B extraction, extended-source extraction, multiple apertures, spectral shifting,
  pruning, shape correction, or xcleanspec behavior in v0.1.
- Do not implement "improved" apertures/extraction until SpeXTool-equivalent regression tests pass.
- Setup may propose associations; reduction must never silently guess an unresolved association.
- B-beam object/standard frames are ignored in the v0.1 NIR workflow.

## Scientific implementation rules

- Reimplement algorithms idiomatically with NumPy/SciPy/Astropy; do not translate IDL syntax
  mechanically.
- Preserve floating-point precision unless a reference comparison demonstrates that lower precision
  is required for compatibility.
- Propagate uncertainties and bit masks explicitly.
- Keep detector correction, calibration, extraction, combination, telluric, and merging operations
  as independently testable functions.
- Every public scientific function should state the SpeXTool routine(s) used as behavioral reference.
- Do not change numerical defaults without updating `docs/PIPELINE_SPEC.md` and tests.
- Do not copy substantial SpeXTool source text verbatim into shuck source or documentation.

## Data and licensing rules

- Do not commit the SpeXTool source tree.
- Do not commit full raw iSHELL data or full SpeXTool calibration/proc products.
- Do not vendor SpeXTool calibration assets until redistribution rights have been established.
- Local development may use `SPEXTOOL5_DIR` to locate SpeXTool 5.0.3 and
  `SHUCK_REFERENCE_DATA` to locate the private regression dataset.
- Unit tests must run without either environment variable. Full regression tests may skip clearly
  when `SHUCK_REFERENCE_DATA` is unavailable.

## CLI contract

Routine user workflow:

```bash
shuck setup NIGHT_DIRECTORY --write-log
# review/edit NIGHT_DIRECTORY/NIGHT.overrides.toml
shuck setup NIGHT_DIRECTORY
shuckit FILE.shuck
```

`shuck setup NIGHT_DIRECTORY/raw` is equivalent. `-o FILE.shuck` optionally overrides the default
`NIGHT_DIRECTORY/NIGHT_DIRECTORY.shuck` output path. Setup automatically discovers the persistent
night-level override TOML; `--overrides` selects another path, and `--write-log` writes the derived
complete FITS-header CSV.

Developer/power-user commands:

```bash
shuck calibrate FILE.shuck
shuck extract FILE.shuck
shuck combine FILE.shuck
shuck telluric FILE.shuck
shuck merge FILE.shuck
```

Validation is automatic at the start of every reduction command. `shuckit` runs all automatic
stages in dependency order and stops on fatal preflight errors.

## Code organization

- `src/shuck/control.py`: parsing and validation of `.shuck` files.
- `src/shuck/io.py`: iSHELL FITS/header I/O only.
- `src/shuck/detector.py`: detector-level corrections and variance/mask construction.
- `src/shuck/calibration/`: dark, flat, wavecal, rectification.
- `src/shuck/extraction/`: spatial profile, aperture finding, tracing, optimal extraction.
- `src/shuck/combine.py`: xcombspec-equivalent scaling and combination.
- `src/shuck/telluric.py`: xtellcor IP/Vega path only.
- `src/shuck/merge.py`: xmergeorders-equivalent behavior.
- `src/shuck/qa.py`: non-interactive diagnostics.
- `src/shuck/provenance.py`: reproducibility metadata.

Avoid GUI code and global mutable state.

## Testing requirements

For every implementation task:

```bash
ruff check .
ruff format --check .
pytest -m "not regression"
```

When the task changes a numerical stage and local reference data are available, also run the
stage-specific regression test(s). Report comparison metrics, not just pass/fail.

Do not weaken tolerances merely to make a regression test pass. Investigate the discrepancy first.
