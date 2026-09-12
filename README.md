# shuck

`shuck` is a Python reduction pipeline for a deliberately narrow subset of NASA IRTF/iSHELL data.
The first implementation targets point-source observations in the **J3** and **Kgas** modes using
the reduction path currently performed with SpeXTool 5.0.3.

The intended routine workflow is:

```bash
shuck setup /path/to/260406 --write-log
# review/edit /path/to/260406/260406.overrides.toml
shuck setup /path/to/260406
shuckit /path/to/260406/260406.shuck
```

The equivalent `shuck setup /path/to/260406/raw` form is also supported. The existing `raw/`
directory contains the inputs; setup creates sibling `cal/`, `proc/`, and `qa/` directories. Use
`-o FILE.shuck` to override the default control-file location. The first run also writes a proposed
persistent override file; reviewed overrides are automatically applied on later runs without
modifying raw FITS headers. `--write-log` optionally writes a complete derived FITS-header CSV at
the night level and leaves any observatory `obslog.txt` untouched.

Developer/power-user stage commands are also planned:

```bash
shuck calibrate night.shuck
shuck extract night.shuck
shuck combine night.shuck
shuck telluric night.shuck
shuck merge night.shuck
```

## Status

Early porting scaffold. Scientific algorithms are not yet implemented.

## Development principles

- Reproduce the selected SpeXTool 5.0.3 scientific behavior before adding improvements.
- Support only J3/Kgas, one-aperture point sources, and A-Sky/Dark in v0.1.
- Keep all science/calibration associations explicit in the `.shuck` control file.
- Test every major numerical stage against a known SpeXTool reduction.
- Do not vendor the SpeXTool distribution or full raw/reference FITS data into this repository.

See `AGENTS.md` and `docs/` for the implementation contract.

## Licensing note

A project license is intentionally not selected in this scaffold. Before public release, determine
what redistribution permissions apply to SpeXTool source-derived algorithms and packaged calibration
data, and separately choose a license for original `shuck` code.
