"""iSHELL FITS input/output helpers."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import date, time
from numbers import Number
from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from astropy.io import fits

from shuck.detector import initial_uncorrected_variance, initialize_detector_mask

FITS_SUFFIXES = frozenset({".fit", ".fits", ".fts"})
COMMENTARY_KEYWORDS = frozenset({"", "COMMENT", "HISTORY", "CONTINUE"})
OBSLOG_PRIORITY_COLUMNS = ("OBJECT", "TCS_OBJ", "DATATYPE", "ITIME", "XDTILT", "BEAM")
ISHELL_DETECTOR_SHAPE = (2048, 2048)
ISHELL_RAW_IMAGE_UNIT = "DN"
ISHELL_RAW_VARIANCE_UNIT = "DN2"


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


@dataclass(frozen=True)
class IShellRawMetadata:
    """Typed primary-header metadata required for one raw iSHELL exposure."""

    itime: float
    coadds: int
    ndr: int
    table_se: float
    divisor: float
    mode: str
    filename: str
    date_obs: date
    time_obs: time
    mjd_obs: float
    object_name: str
    beam: str
    ra: str
    dec: str
    airmass: float
    hour_angle: str
    position_angle: float
    slit_width_arcsec: float | None = None


@dataclass(frozen=True)
class RawIShellFrame:
    """One decoded native iSHELL exposure in detector-native orientation.

    All detector images are independent float64 arrays in NumPy
    ``(row, column)`` order and units of :attr:`image_unit`. ``raw_image``
    is reconstructed from ``pedestal_image - signal_image``. The native
    primary difference image is retained separately but is not required or
    validated to equal that reconstruction. ``variance`` describes
    ``raw_image`` and has units of :attr:`variance_unit`.

    The behavioral reference for the MEF mapping and reconstruction is
    SpeXTool 5.0.3
    ``instruments/ishell/pro/mc_readishellfits.pro``.
    """

    path: Path
    raw_image: np.ndarray
    primary_difference_image: np.ndarray
    pedestal_image: np.ndarray
    signal_image: np.ndarray
    header: fits.Header
    metadata: IShellRawMetadata
    variance: np.ndarray
    mask: np.ndarray
    image_unit: str = ISHELL_RAW_IMAGE_UNIT
    variance_unit: str = ISHELL_RAW_VARIANCE_UNIT


class RawIShellFitsError(ValueError):
    """Raised when a FITS file is not a valid native iSHELL raw exposure."""


def read_ishell_raw_metadata(path: str | Path) -> IShellRawMetadata:
    """Read only metadata required downstream from an iSHELL primary header."""

    frame_path = Path(path)
    header = fits.getheader(frame_path, ext=0)
    return _parse_ishell_raw_metadata(header, frame_path)


def read_ishell_raw(path: str | Path) -> RawIShellFrame:
    """Read one native iSHELL MEF exposure into a DN-valued raw frame.

    The reader maps HDU 0 to the native primary difference image, HDU 1 to
    the summed pedestal reads, and HDU 2 to the summed signal reads. It
    applies the primary-header ``DIVISOR`` once to each stored array and uses
    pedestal minus signal as the canonical ``raw_image``. Arrays remain in
    detector-native NumPy ``(row, column)`` orientation; no transpose,
    rotation, or flip is performed.

    The behavioral reference is SpeXTool 5.0.3
    ``instruments/ishell/pro/mc_readishellfits.pro``. Unlike that broader
    routine, this raw-reader boundary does not normalize by integration time
    or apply detector corrections, pairing, or calibration.
    """

    frame_path = Path(path)
    with fits.open(frame_path, mode="readonly", memmap=False) as hdus:
        _validate_raw_hdus(hdus, frame_path)
        header = hdus[0].header.copy()
        metadata = _parse_ishell_raw_metadata(header, frame_path)
        decoded_arrays = []
        for hdu in hdus:
            decoded = np.array(hdu.data, dtype=np.float64, copy=True)
            decoded /= metadata.divisor
            decoded_arrays.append(decoded)

    primary_difference_image, pedestal_image, signal_image = decoded_arrays
    raw_image = pedestal_image - signal_image
    variance = initial_uncorrected_variance(
        raw_image,
        itime=metadata.itime,
        coadds=metadata.coadds,
        ndr=metadata.ndr,
        table_se=metadata.table_se,
        divisor=metadata.divisor,
    )
    mask = initialize_detector_mask(raw_image.shape)

    return RawIShellFrame(
        path=frame_path,
        raw_image=raw_image,
        primary_difference_image=primary_difference_image,
        pedestal_image=pedestal_image,
        signal_image=signal_image,
        header=header,
        metadata=metadata,
        variance=variance,
        mask=mask,
    )


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


def _validate_raw_hdus(hdus: fits.HDUList, path: Path) -> None:
    if len(hdus) != 3:
        raise RawIShellFitsError(
            f"Expected exactly 3 HDUs in native iSHELL FITS file {path}; found {len(hdus)}"
        )
    if not isinstance(hdus[0], fits.PrimaryHDU):
        raise RawIShellFitsError(f"HDU 0 is not a primary image HDU in {path}")
    for index, hdu in enumerate(hdus):
        if not isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU)) or hdu.data is None:
            raise RawIShellFitsError(f"HDU {index} is not a populated image HDU in {path}")
        if hdu.data.shape != ISHELL_DETECTOR_SHAPE:
            raise RawIShellFitsError(
                f"HDU {index} in {path} has detector shape {hdu.data.shape}; "
                f"expected {ISHELL_DETECTOR_SHAPE} in (row, column) order"
            )
        if not np.issubdtype(hdu.data.dtype, np.number):
            raise RawIShellFitsError(
                f"HDU {index} does not contain numeric detector data in {path}"
            )


def _parse_ishell_raw_metadata(header: fits.Header, path: Path) -> IShellRawMetadata:
    instrument = _required_value(header, "INSTRUME", str, path)
    if instrument.casefold() != "ishell spectrograph":
        raise RawIShellFitsError(
            f"Invalid INSTRUME={instrument!r} in {path}; expected 'iSHELL Spectrograph'"
        )

    itime = _required_positive_float(header, "ITIME", path)
    coadds = _required_positive_int(header, "CO_ADDS", path)
    ndr = _required_positive_int(header, "NDR", path)
    table_se = _required_positive_float(header, "TABLE_SE", path)
    divisor = _required_positive_float(header, "DIVISOR", path)
    mode = _required_value(header, "XDTILT", str, path)
    filename = _required_value(header, "IRAFNAME", str, path)
    object_name = _required_value(header, "OBJECT", str, path)
    beam = _required_value(header, "BEAM", str, path).upper()
    if beam not in {"A", "B"}:
        raise RawIShellFitsError(f"Invalid BEAM={beam!r} in {path}; expected 'A' or 'B'")

    date_text = _required_value(header, "DATE_OBS", str, path)
    time_text = _required_value(header, "TIME_OBS", str, path)
    try:
        date_obs = date.fromisoformat(date_text)
    except ValueError as error:
        raise RawIShellFitsError(f"Invalid DATE_OBS={date_text!r} in {path}") from error
    try:
        time_obs = time.fromisoformat(time_text)
    except ValueError as error:
        raise RawIShellFitsError(f"Invalid TIME_OBS={time_text!r} in {path}") from error

    mjd_obs = _required_finite_float(header, "MJD_OBS", path)
    airmass = _required_positive_float(header, "TCS_AM", path)
    position_angle = _required_finite_float(header, "POSANGLE", path)
    slit_value = header.get("SLIT")
    try:
        slit_width_arcsec = None if slit_value is None else float(slit_value)
    except (TypeError, ValueError):
        # Calibration frames conventionally use SLIT='Mirror'. Downstream
        # telluric processing explicitly requires a numeric science slit.
        slit_width_arcsec = None
    if slit_width_arcsec is not None and (
        not np.isfinite(slit_width_arcsec) or slit_width_arcsec <= 0
    ):
        raise RawIShellFitsError(
            f"Invalid SLIT={slit_width_arcsec!r} in {path}; expected value > 0"
        )
    ra = _required_value(header, "TCS_RA", str, path)
    dec = _required_value(header, "TCS_DEC", str, path)
    hour_angle = _required_value(header, "TCS_HA", str, path)
    _validate_angle(ra, "TCS_RA", u.hourangle, path)
    _validate_angle(dec, "TCS_DEC", u.deg, path)
    _validate_angle(hour_angle, "TCS_HA", u.hourangle, path)

    return IShellRawMetadata(
        itime=itime,
        coadds=coadds,
        ndr=ndr,
        table_se=table_se,
        divisor=divisor,
        mode=mode,
        filename=filename,
        date_obs=date_obs,
        time_obs=time_obs,
        mjd_obs=mjd_obs,
        object_name=object_name,
        beam=beam,
        ra=ra,
        dec=dec,
        airmass=airmass,
        hour_angle=hour_angle,
        position_angle=position_angle,
        slit_width_arcsec=slit_width_arcsec,
    )


def _required_value(header: fits.Header, keyword: str, converter: type[Any], path: Path) -> Any:
    try:
        value = _optional_value(header, keyword, converter)
    except ValueError as error:
        raise RawIShellFitsError(f"{error} in {path}") from error
    if value is None:
        raise RawIShellFitsError(f"Missing required {keyword} in {path}")
    return value


def _required_finite_float(header: fits.Header, keyword: str, path: Path) -> float:
    value = _required_value(header, keyword, float, path)
    if not np.isfinite(value):
        raise RawIShellFitsError(f"Invalid {keyword}={value!r} in {path}; expected finite value")
    return value


def _required_positive_float(header: fits.Header, keyword: str, path: Path) -> float:
    value = _required_finite_float(header, keyword, path)
    if value <= 0.0:
        raise RawIShellFitsError(f"Invalid {keyword}={value!r} in {path}; expected value > 0")
    return value


def _required_positive_int(header: fits.Header, keyword: str, path: Path) -> int:
    raw_value = _required_value(header, keyword, float, path)
    if not np.isfinite(raw_value) or raw_value <= 0.0 or not raw_value.is_integer():
        raise RawIShellFitsError(
            f"Invalid {keyword}={raw_value!r} in {path}; expected a positive integer"
        )
    return int(raw_value)


def _validate_angle(value: str, keyword: str, unit: u.UnitBase, path: Path) -> None:
    try:
        angle = Angle(value, unit=unit)
    except (TypeError, ValueError) as error:
        raise RawIShellFitsError(f"Invalid {keyword}={value!r} in {path}") from error
    if not np.isfinite(angle.degree):
        raise RawIShellFitsError(f"Invalid {keyword}={value!r} in {path}")


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
