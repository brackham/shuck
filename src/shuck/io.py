"""iSHELL FITS input/output helpers."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Any

from astropy.io import fits

FITS_SUFFIXES = frozenset({".fit", ".fits", ".fts"})
COMMENTARY_KEYWORDS = frozenset({"", "COMMENT", "HISTORY", "CONTINUE"})
OBSLOG_PRIORITY_COLUMNS = ("OBJECT", "TCS_OBJ", "DATATYPE", "ITIME", "XDTILT", "BEAM")


@dataclass(frozen=True)
class RawFrameHeader:
    """Header metadata needed to construct and validate a v1 control file."""

    path: Path
    instrument: str | None
    object_name: str | None
    datatype: str | None
    mode: str | None
    beam: str | None
    slit: str | None
    itime: float | None
    ndr: int | None
    coadds: int | None
    mjd_obs: float | None
    qth_lamp: str | None
    ir_lamp: str | None
    arg_lamp: str | None
    window_cover: str | None
    calibration_mirror: str | None
    arc_stage: str | None
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class NightLayout:
    """Resolved standard v0.1 night and raw-data directories."""

    night_directory: Path
    raw_directory: Path


@dataclass(frozen=True)
class FitsHeaderRecord:
    """Scalar FITS-header values for one raw file."""

    filename: str
    values: tuple[tuple[str, str], ...]
    mjd_obs: float | None


def scan_fits_headers(raw_directory: str | Path) -> tuple[RawFrameHeader, ...]:
    """Read primary-header metadata from FITS files directly in ``raw_directory``.

    This function performs header I/O only; classification belongs in
    :mod:`shuck.control`.
    """

    directory = Path(raw_directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Raw-data directory does not exist: {directory}")

    paths = _fits_paths(directory)
    headers = tuple(read_fits_header(path) for path in paths)
    if headers and all(header.mjd_obs is not None for header in headers):
        return tuple(sorted(headers, key=lambda header: (header.mjd_obs, header.path.name)))
    return headers


def resolve_night_layout(directory: str | Path) -> NightLayout:
    """Resolve the standard night layout from a night directory or its ``raw/`` child."""

    requested = Path(directory).resolve()
    if requested.name.casefold() == "raw":
        night_directory = requested.parent
        raw_directory = requested
    else:
        night_directory = requested
        raw_directory = requested / "raw"

    guidance = "Pass either a night directory containing raw/ or the raw/ directory itself."
    if not night_directory.is_dir():
        raise NotADirectoryError(f"Night directory does not exist: {night_directory}. {guidance}")
    if not raw_directory.is_dir():
        raise FileNotFoundError(
            f"Required raw-data directory does not exist: {raw_directory}. {guidance} "
            "Setup did not create it."
        )
    if not _contains_usable_ishell_fits(_fits_paths(raw_directory)):
        raise FileNotFoundError(
            f"No usable iSHELL spectrograph FITS files were found in raw-data directory: "
            f"{raw_directory}. {guidance}"
        )
    return NightLayout(night_directory=night_directory, raw_directory=raw_directory)


def read_fits_header(path: str | Path) -> RawFrameHeader:
    """Read the setup-relevant metadata from one iSHELL FITS primary header."""

    frame_path = Path(path)
    header = fits.getheader(frame_path, ext=0)
    issues: list[str] = []

    datatype = _optional_value(header, "DATATYPE", str)
    # During iSHELL calibration exposures TCS_OBJ normally retains a science
    # target while OBJECT identifies the lamp or dark role.  That is expected
    # metadata, not an object-identity conflict.  Target exposures retain the
    # stricter alias check used for reviewable identity disagreements.
    if _casefold(datatype) == "calibration":
        object_name = _optional_value(header, "OBJECT", str) or _optional_value(
            header, "TCS_OBJ", str
        )
    else:
        object_name = _matching_alias_value(
            header, ("OBJECT", "TCS_OBJ"), str, "object identity", issues
        )
    ndr = _matching_alias_value(header, ("NDR", "IA_NDR"), int, "NDR", issues)
    coadds = _matching_alias_value(header, ("CO_ADDS", "IA_COADD"), int, "coadds", issues)
    arg_lamp = _matching_alias_value(
        header, ("ARG_LAMP", "ARGLAMP"), str, "ThAr lamp state", issues
    )
    return RawFrameHeader(
        path=frame_path,
        instrument=_optional_value(header, "INSTRUME", str),
        object_name=object_name,
        datatype=datatype,
        mode=_optional_value(header, "XDTILT", str),
        beam=_optional_value(header, "BEAM", str),
        slit=_optional_value(header, "SLIT", str),
        itime=_optional_value(header, "ITIME", float),
        ndr=ndr,
        coadds=coadds,
        mjd_obs=_optional_value(header, "MJD_OBS", float),
        qth_lamp=_optional_value(header, "QTH_LAMP", str),
        ir_lamp=_optional_value(header, "IR_LAMP", str),
        arg_lamp=arg_lamp,
        window_cover=_optional_value(header, "WINCOV", str),
        calibration_mirror=_optional_value(header, "CALMIR", str),
        arc_stage=_optional_value(header, "ARLMPSTG", str),
        issues=tuple(issues),
    )


def scan_fits_header_inventory(raw_directory: str | Path) -> tuple[FitsHeaderRecord, ...]:
    """Read all representable scalar header cards from each FITS file in a raw directory."""

    directory = Path(raw_directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Raw-data directory does not exist: {directory}")

    records = tuple(_read_fits_header_record(path) for path in _fits_paths(directory))
    if records and all(record.mjd_obs is not None for record in records):
        return tuple(sorted(records, key=lambda record: (record.mjd_obs, record.filename)))
    return records


def write_fits_observation_log(raw_directory: str | Path, output_path: str | Path) -> None:
    """Write a deterministic complete scalar-header inventory as CSV."""

    records = scan_fits_header_inventory(raw_directory)
    columns: list[str] = []
    seen: set[str] = set()
    for record in records:
        for keyword, _ in record.values:
            if keyword not in seen:
                columns.append(keyword)
                seen.add(keyword)
    columns = [
        *(keyword for keyword in OBSLOG_PRIORITY_COLUMNS if keyword in seen),
        *(keyword for keyword in columns if keyword not in OBSLOG_PRIORITY_COLUMNS),
    ]

    with Path(output_path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["filename", *columns], lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({"filename": record.filename, **dict(record.values)})


def _read_fits_header_record(path: Path) -> FitsHeaderRecord:
    values: list[tuple[str, str]] = []
    occurrences: dict[str, int] = {}
    mjd_obs: float | None = None
    with fits.open(path, mode="readonly", memmap=False) as hdus:
        for hdu_index, hdu in enumerate(hdus):
            for card in hdu.header.cards:
                keyword = card.keyword.strip()
                if keyword.upper() in COMMENTARY_KEYWORDS or not _is_scalar_header_value(
                    card.value
                ):
                    continue
                base_name = keyword if hdu_index == 0 else f"HDU{hdu_index}.{keyword}"
                occurrences[base_name] = occurrences.get(base_name, 0) + 1
                occurrence = occurrences[base_name]
                column_name = base_name if occurrence == 1 else f"{base_name}#{occurrence}"
                values.append((column_name, str(card.value)))
                if hdu_index == 0 and keyword.upper() == "MJD_OBS" and mjd_obs is None:
                    try:
                        mjd_obs = float(card.value)
                    except (TypeError, ValueError):
                        pass
    return FitsHeaderRecord(path.name, tuple(values), mjd_obs)


def _is_scalar_header_value(value: object) -> bool:
    return isinstance(value, (str, Number))


def _optional_value(header: fits.Header, keyword: str, converter: type[Any]) -> Any | None:
    value = header.get(keyword)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        converted = converter(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {keyword}={value!r}") from error
    return converted.strip() if isinstance(converted, str) else converted


def _casefold(value: str | None) -> str:
    return value.strip().casefold() if value else ""


def _matching_alias_value(
    header: fits.Header,
    keywords: tuple[str, ...],
    converter: type[Any],
    label: str,
    issues: list[str],
) -> Any | None:
    values = [
        (keyword, value)
        for keyword in keywords
        if (value := _optional_value(header, keyword, converter)) is not None
    ]
    if not values:
        return None

    first_value = values[0][1]
    if any(value != first_value for _, value in values[1:]):
        rendered = ", ".join(f"{keyword}={value!r}" for keyword, value in values)
        issues.append(f"conflicting {label} keywords ({rendered})")
        return None
    return first_value


def _natural_name_key(value: str) -> tuple[tuple[int, int | str], ...]:
    """Return a deterministic file-order key with numeric components sorted numerically."""

    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", value)
        if part
    )


def _fits_paths(directory: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.casefold() in FITS_SUFFIXES
            ),
            key=lambda path: _natural_name_key(path.name),
        )
    )


def _contains_usable_ishell_fits(paths: tuple[Path, ...]) -> bool:
    for path in paths:
        try:
            instrument = fits.getheader(path, ext=0).get("INSTRUME")
        except OSError:
            continue
        if isinstance(instrument, str) and instrument.strip().casefold() == "ishell spectrograph":
            return True
    return False
