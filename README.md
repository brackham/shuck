# shuck

`shuck` is a Python reduction pipeline for a deliberately narrow subset of NASA IRTF/iSHELL data.
The first implementation targets point-source observations in the **J3** and **Kgas** modes using
the reduction path currently performed with SpeXTool 5.0.3.

The intended routine workflow is:

```bash
shuck setup /path/to/night/raw -o night.shuck
# inspect and edit night.shuck
shuckit night.shuck
```

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
