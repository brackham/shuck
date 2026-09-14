"""Package-native access to bundled scientific reference assets."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

SPEXTOOL_RESOURCE_PACKAGE = "shuck.data.spextool"

REQUIRED_SPEXTOOL_ASSETS = (
    "data/version.dat",
    "data/vega500000.sav",
    "instruments/ishell/data/ishell.dat",
    "instruments/ishell/data/xtellcor_modeinfo.dat",
    "instruments/ishell/data/IP_coefficients.dat",
    "instruments/ishell/data/ishell_bias.fits",
    "instruments/ishell/data/ishell_lincorr_CDS.fits",
    "instruments/ishell/data/ishell_bdpxmk.fits",
    "instruments/ishell/data/ishell_htpxmk.fits",
    "instruments/ishell/data/J3_flatinfo.fits",
    "instruments/ishell/data/J3_wavecalinfo.fits",
    "instruments/ishell/data/J3_lines.dat",
    "instruments/ishell/data/Kgas_flatinfo.fits",
    "instruments/ishell/data/Kgas_wavecalinfo.fits",
    "instruments/ishell/data/Kgas_lines.dat",
)


def spextool_root() -> Path:
    """Return the installed Spextool 5.0.3 resource tree.

    The resources are ordinary package data in wheels and installed source
    distributions.  ``importlib.resources`` avoids assumptions about the
    repository location or current working directory.
    """

    resource = files(SPEXTOOL_RESOURCE_PACKAGE)
    try:
        root = Path(resource)
    except TypeError as exception:  # pragma: no cover - normal installs are unpacked
        raise RuntimeError(
            "shuck's bundled Spextool assets require a normal unpacked Python installation"
        ) from exception
    missing = [relative for relative in REQUIRED_SPEXTOOL_ASSETS if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            "shuck's installed Spextool reference data are incomplete: " + ", ".join(missing)
        )
    return root
