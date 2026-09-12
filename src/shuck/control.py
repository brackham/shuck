"""Parsing, setup generation, and preflight validation for ``.shuck`` files."""

from __future__ import annotations

import configparser
import json
import os
import re
import tomllib
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

from shuck.io import (
    RawFrameHeader,
    read_fits_header,
    resolve_night_layout,
    scan_fits_headers,
)

SUPPORTED_MODES = frozenset({"J3", "Kgas"})
TABLE_COLUMNS = {
    "calibrations": (
        "calib_id",
        "mode",
        "flat_files",
        "arc_on_files",
        "arc_off_files",
    ),
    "darks": ("dark_id", "itime", "ndr", "coadds", "files"),
    "standards": ("standard_id", "target", "bmag", "vmag", "rv_kms"),
    "data": (
        "filename",
        "frametype",
        "target",
        "mode",
        "beam",
        "calib",
        "dark",
        "comb_id",
        "telluric_group",
    ),
}
# v0.1 policy: fewer than five dark frames is advisory only and does not alter
# any numerical reduction behavior.
SMALL_DARK_GROUP_SIZE = 5


class ControlFileError(ValueError):
    """A syntax or type error in a ``.shuck`` control file."""


@dataclass(frozen=True)
class Placeholder:
    """An explicitly unresolved, human-editable control-file value."""

    text: str

    def __str__(self) -> str:
        return f"<{self.text}>"


FileReference = str | Placeholder
NumericValue = float | Placeholder


@dataclass(frozen=True)
class ShuckSettings:
    format_version: int
    raw_path: Path
    calib_dir: Path
    proc_dir: Path
    qa_dir: Path


@dataclass(frozen=True)
class ExtractionSettings:
    reduction_mode: str
    n_apertures: int
    optimal: bool
    psf_radius: float
    aperture_radius: float
    background_start: float
    background_width: float
    background_degree: int
    trace_degree: int


@dataclass(frozen=True)
class CombineSettings:
    statistic: str
    sigma_clip: float
    j3_scale_order: int
    kgas_scale_order: int
    shift_spectra: bool
    prune: bool
    shape_correction: bool


@dataclass(frozen=True)
class TelluricSettings:
    method: str
    find_shifts: bool


@dataclass(frozen=True)
class CalibrationGroup:
    calib_id: str
    mode: str
    flat_files: tuple[FileReference, ...]
    arc_on_files: tuple[FileReference, ...]
    arc_off_files: tuple[FileReference, ...]


@dataclass(frozen=True)
class DarkGroup:
    dark_id: str
    itime: float
    ndr: int
    coadds: int
    files: tuple[FileReference, ...]


@dataclass(frozen=True)
class Standard:
    standard_id: str
    target: str
    bmag: NumericValue
    vmag: NumericValue
    rv_kms: NumericValue


class FrameType(StrEnum):
    FLAT = "flat"
    ARC_ON = "arc_on"
    ARC_OFF = "arc_off"
    DARK = "dark"
    SCIENCE = "science"
    STANDARD = "standard"
    IGNORE = "ignore"


@dataclass(frozen=True)
class DataFrame:
    filename: FileReference
    frametype: FrameType
    target: str | None
    mode: str | None
    beam: str | None
    calib: str | None
    dark: str | None
    comb_id: str | None
    telluric_group: str | None


@dataclass(frozen=True)
class ControlFile:
    shuck: ShuckSettings
    extraction: ExtractionSettings
    combine: CombineSettings
    telluric: TelluricSettings
    calibrations: tuple[CalibrationGroup, ...]
    darks: tuple[DarkGroup, ...]
    standards: tuple[Standard, ...]
    data: tuple[DataFrame, ...]
    source_path: Path | None = None


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class Diagnostic:
    severity: Severity
    code: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    diagnostics: tuple[Diagnostic, ...]

    @property
    def errors(self) -> tuple[Diagnostic, ...]:
        return tuple(item for item in self.diagnostics if item.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[Diagnostic, ...]:
        return tuple(item for item in self.diagnostics if item.severity is Severity.WARNING)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class SetupResult:
    control: ControlFile
    review_notes: tuple[str, ...]
    night_directory: Path
    raw_directory: Path
    output_path: Path
    overrides_path: Path
    proposed_overrides: str | None


class RawFrameKind(StrEnum):
    FLAT = "flat"
    ARC_ON = "arc-on"
    ARC_OFF = "arc-off"
    DARK = "dark"
    SCIENCE = "science"
    STANDARD = "standard"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class ClassifiedFrame:
    header: RawFrameHeader
    kind: RawFrameKind
    reasons: tuple[str, ...] = ()
    object_source: str = "header"
    frametype_override: FrameType | None = None
    frametype_source: str = "header/inference"


@dataclass(frozen=True)
class ObjectOverride:
    """One exact object-name override from a setup override file."""

    object_name: str
    frametype: FrameType


@dataclass(frozen=True)
class FileOverride:
    """One exact filename override from a setup override file."""

    filename: str
    object_name: str | None
    frametype: FrameType | None


@dataclass(frozen=True)
class SetupOverrides:
    """Parsed v0.1 setup overrides."""

    objects: tuple[ObjectOverride, ...] = ()
    files: tuple[FileOverride, ...] = ()


def parse_control_file(path: str | Path) -> ControlFile:
    """Parse a format-version-1 ``.shuck`` file into a typed model."""

    source_path = Path(path)
    try:
        text = source_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ControlFileError(f"Could not read control file {source_path}: {error}") from error
    return parse_control_text(text, source_path=source_path)


def parse_control_text(text: str, *, source_path: str | Path | None = None) -> ControlFile:
    """Parse format-version-1 control-file text into a typed model."""

    config_lines, tables = _split_config_and_tables(text)
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read_string("\n".join(config_lines))
    except configparser.Error as error:
        raise ControlFileError(f"Invalid configuration syntax: {error}") from error

    _require_sections(parser, ("shuck", "extraction", "combine"))
    shuck = ShuckSettings(
        format_version=_config_int(parser, "shuck", "format_version"),
        raw_path=Path(_config_string(parser, "shuck", "raw_path")),
        calib_dir=Path(_config_string(parser, "shuck", "calib_dir")),
        proc_dir=Path(_config_string(parser, "shuck", "proc_dir")),
        qa_dir=Path(_config_string(parser, "shuck", "qa_dir")),
    )
    if shuck.format_version != 1:
        raise ControlFileError(
            f"Unsupported [shuck] format_version={shuck.format_version}; expected 1"
        )
    extraction = ExtractionSettings(
        reduction_mode=_config_string(parser, "extraction", "reduction_mode"),
        n_apertures=_config_int(parser, "extraction", "n_apertures"),
        optimal=_config_bool(parser, "extraction", "optimal"),
        psf_radius=_config_float(parser, "extraction", "psf_radius"),
        aperture_radius=_config_float(parser, "extraction", "aperture_radius"),
        background_start=_config_float(parser, "extraction", "background_start"),
        background_width=_config_float(parser, "extraction", "background_width"),
        background_degree=_config_int(parser, "extraction", "background_degree"),
        trace_degree=_config_int(parser, "extraction", "trace_degree"),
    )
    combine = CombineSettings(
        statistic=_config_string(parser, "combine", "statistic"),
        sigma_clip=_config_float(parser, "combine", "sigma_clip"),
        j3_scale_order=_config_int(parser, "combine", "J3_scale_order"),
        kgas_scale_order=_config_int(parser, "combine", "Kgas_scale_order"),
        shift_spectra=_config_bool(parser, "combine", "shift_spectra"),
        prune=_config_bool(parser, "combine", "prune"),
        shape_correction=_config_bool(parser, "combine", "shape_correction"),
    )
    telluric = (
        TelluricSettings(
            method=_config_string(parser, "telluric", "method"),
            find_shifts=_config_bool(parser, "telluric", "find_shifts"),
        )
        if parser.has_section("telluric")
        else TelluricSettings(method="IP", find_shifts=False)
    )
    parsed = {name: _parse_table(name, tables[name]) for name in TABLE_COLUMNS}
    return ControlFile(
        shuck=shuck,
        extraction=extraction,
        combine=combine,
        telluric=telluric,
        calibrations=tuple(_parse_calibration(row) for row in parsed["calibrations"]),
        darks=tuple(_parse_dark(row) for row in parsed["darks"]),
        standards=tuple(_parse_standard(row) for row in parsed["standards"]),
        data=tuple(_parse_data_frame(row) for row in parsed["data"]),
        source_path=Path(source_path) if source_path is not None else None,
    )


def parse_setup_overrides(path: str | Path) -> SetupOverrides:
    """Parse the small, exact-match v0.1 setup override schema."""

    override_path = Path(path)
    try:
        with override_path.open("rb") as stream:
            document = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ControlFileError(f"Could not parse override file {override_path}: {error}") from error

    unknown_sections = set(document).difference({"objects", "files"})
    if unknown_sections:
        names = ", ".join(sorted(unknown_sections))
        raise ControlFileError(f"Unknown top-level override section(s): {names}")

    object_entries = _override_table(document, "objects")
    file_entries = _override_table(document, "files")
    objects: list[ObjectOverride] = []
    for object_name, values in object_entries.items():
        _reject_override_fields(values, {"frametype"}, f"object {object_name!r}")
        frametype = _override_frametype(values, f"object {object_name!r}")
        if frametype not in {FrameType.SCIENCE, FrameType.STANDARD, FrameType.IGNORE}:
            raise ControlFileError(
                f"Override object {object_name!r} has frametype {frametype.value!r}; "
                "object rules may only classify target exposures"
            )
        objects.append(
            ObjectOverride(
                object_name=object_name,
                frametype=frametype,
            )
        )

    files: list[FileOverride] = []
    for filename, values in file_entries.items():
        _reject_override_fields(values, {"object", "frametype"}, f"file {filename!r}")
        object_name = values.get("object")
        if object_name is not None and (
            not isinstance(object_name, str) or not object_name.strip()
        ):
            raise ControlFileError(f"Override file {filename!r} object must be a nonblank string")
        frametype = (
            _override_frametype(values, f"file {filename!r}") if "frametype" in values else None
        )
        if object_name is None and frametype is None:
            raise ControlFileError(
                f"Override file {filename!r} must define object and/or frametype"
            )
        files.append(
            FileOverride(filename, object_name.strip() if object_name else None, frametype)
        )

    return SetupOverrides(
        objects=tuple(sorted(objects, key=lambda item: item.object_name.casefold())),
        files=tuple(sorted(files, key=lambda item: item.filename.casefold())),
    )


def classify_raw_frame(header: RawFrameHeader) -> ClassifiedFrame:
    """Conservatively classify one iSHELL header for setup generation."""

    reasons = list(header.issues)
    if _casefold(header.instrument) != "ishell spectrograph":
        reasons.append(f"unsupported or missing INSTRUME={header.instrument!r}")
        return ClassifiedFrame(header, RawFrameKind.AMBIGUOUS, tuple(reasons))
    datatype = _casefold(header.datatype)
    object_name = _casefold(header.object_name)
    filename = header.path.name.casefold()
    mode = _casefold(header.mode)
    qth_on = _state_is(header.qth_lamp, "on")
    ir_on = _state_is(header.ir_lamp, "on")
    arg_on = _state_is(header.arg_lamp, "on")
    arg_off = _state_is(header.arg_lamp, "off")
    flat_name = _contains_word(object_name, "flat") or _contains_word(filename, "flat")
    dark_name = _contains_word(object_name, "dark") or _contains_word(filename, "dark")
    arc_name = any(
        _contains_word(value, token)
        for value in (object_name, filename)
        for token in ("arc", "thar")
    )
    calibration_hardware = qth_on or ir_on or arg_on or mode == "darks"

    if header.issues:
        return ClassifiedFrame(header, RawFrameKind.AMBIGUOUS, tuple(reasons))
    # DATATYPE=calibration and the calibration OBJECT values below are the
    # authoritative iSHELL identifiers.  TCS_OBJ commonly remains set to the
    # preceding target while these frames are taken, so it is intentionally not
    # consulted here.
    if datatype == "calibration":
        if _has_trusted_dark_signature(header) or mode == "darks" or object_name == "dark":
            return ClassifiedFrame(header, RawFrameKind.DARK)
        if object_name == "qth":
            return ClassifiedFrame(header, RawFrameKind.FLAT)
        if object_name == "thar on":
            return ClassifiedFrame(header, RawFrameKind.ARC_ON)
        if object_name == "thar off":
            return ClassifiedFrame(header, RawFrameKind.ARC_OFF)
        # Retain the lamp-state fallbacks for older iSHELL headers, but only
        # after the explicit OBJECT values have been considered.
        if arg_on:
            return ClassifiedFrame(header, RawFrameKind.ARC_ON)
        if qth_on or ir_on:
            return ClassifiedFrame(header, RawFrameKind.FLAT)
        if arc_name and arg_off:
            return ClassifiedFrame(header, RawFrameKind.ARC_OFF)
        reasons.append("calibration exposure has no unambiguous lamp/dark classification")
        return ClassifiedFrame(header, RawFrameKind.AMBIGUOUS, tuple(reasons))
    if datatype in {"target", "standard"} and (
        calibration_hardware or flat_name or dark_name or arc_name
    ):
        reasons.append(f"DATATYPE={header.datatype!r} conflicts with calibration header evidence")
        return ClassifiedFrame(header, RawFrameKind.AMBIGUOUS, tuple(reasons))
    if arg_on and (qth_on or ir_on):
        reasons.append("ThAr and flat lamps are both reported on")
        return ClassifiedFrame(header, RawFrameKind.AMBIGUOUS, tuple(reasons))
    standard_name = any(
        _contains_word(value, token)
        for value in (object_name, filename)
        for token in ("standard", "telluric", "std")
    )
    science_name = any(
        _contains_word(value, token)
        for value in (object_name, filename)
        for token in ("science", "target")
    )
    if datatype == "target":
        if standard_name:
            reasons.append("DATATYPE='target' conflicts with a standard/telluric label")
            return ClassifiedFrame(header, RawFrameKind.AMBIGUOUS, tuple(reasons))
        return ClassifiedFrame(header, RawFrameKind.SCIENCE)
    if datatype == "standard":
        if science_name:
            reasons.append("DATATYPE='standard' conflicts with a science/target label")
            return ClassifiedFrame(header, RawFrameKind.AMBIGUOUS, tuple(reasons))
        return ClassifiedFrame(header, RawFrameKind.STANDARD)
    reasons.append(f"unrecognized or missing DATATYPE={header.datatype!r}")
    return ClassifiedFrame(header, RawFrameKind.AMBIGUOUS, tuple(reasons))


def build_setup_control(
    raw_directory: str | Path,
    *,
    output_path: str | Path | None = None,
    overrides_path: str | Path | None = None,
) -> SetupResult:
    """Inspect a standard night layout and propose an editable format-version-1 model."""

    layout = resolve_night_layout(raw_directory)
    night_directory = layout.night_directory
    raw_path = layout.raw_directory
    calibration_directory = night_directory / "cal"
    processing_directory = night_directory / "proc"
    qa_directory = night_directory / "qa"
    for directory in (calibration_directory, processing_directory, qa_directory):
        directory.mkdir(exist_ok=True)

    base_classified = tuple(classify_raw_frame(header) for header in scan_fits_headers(raw_path))
    review_notes: list[str] = []
    review_notes.extend(_filename_object_disagreement_notes(base_classified))

    resolved_overrides_path = (
        night_directory / f"{night_directory.name}.overrides.toml"
        if overrides_path is None
        else Path(overrides_path).resolve()
    )
    proposed_overrides: str | None = None
    if resolved_overrides_path.exists():
        overrides = parse_setup_overrides(resolved_overrides_path)
        classified = _apply_setup_overrides(base_classified, overrides, review_notes)
        review_notes.extend(_unresolved_classification_notes(classified, overrides))
    else:
        classified = base_classified
        suggestions, suggestion_notes = _classification_override_suggestions(classified)
        review_notes.extend(
            f"{note}; proposed override will be written to {resolved_overrides_path.name}"
            for note in suggestion_notes
        )
        proposed_overrides = _render_override_suggestions(suggestions)

    for frame in classified:
        if (
            frame.kind is RawFrameKind.AMBIGUOUS
            and frame.frametype_override is not FrameType.IGNORE
        ):
            detail = "; ".join(frame.reasons) or "insufficient header evidence"
            review_notes.append(f"{frame.header.path.name}: {detail}")
    calibrations, calibration_positions = _build_calibration_groups(classified, review_notes)
    darks, dark_ids = _build_dark_groups(classified, review_notes)
    standards = _build_standards(classified)
    data = _build_data_rows(
        classified,
        calibrations,
        calibration_positions,
        dark_ids,
        review_notes,
    )

    source_path = (
        night_directory / f"{night_directory.name}.shuck"
        if output_path is None
        else Path(output_path).resolve()
    )

    def relative_to_control(directory: Path) -> Path:
        return Path(os.path.relpath(directory, start=source_path.parent))

    control = ControlFile(
        shuck=ShuckSettings(
            1,
            relative_to_control(raw_path),
            relative_to_control(calibration_directory),
            relative_to_control(processing_directory),
            relative_to_control(qa_directory),
        ),
        extraction=ExtractionSettings("A-Sky/Dark", 1, True, 1.0, 1.0, 1.1, 2.0, 0, 2),
        combine=CombineSettings("robust_weighted_mean", 8.0, 402, 224, False, False, False),
        telluric=TelluricSettings("IP", False),
        calibrations=calibrations,
        darks=darks,
        standards=standards,
        data=data,
        source_path=source_path,
    )
    return SetupResult(
        control=control,
        review_notes=tuple(review_notes),
        night_directory=night_directory,
        raw_directory=raw_path,
        output_path=source_path,
        overrides_path=resolved_overrides_path,
        proposed_overrides=proposed_overrides,
    )


def _apply_setup_overrides(
    frames: tuple[ClassifiedFrame, ...],
    overrides: SetupOverrides,
    review_notes: list[str],
) -> tuple[ClassifiedFrame, ...]:
    object_rules = {item.object_name: item for item in overrides.objects}
    file_rules = {item.filename: item for item in overrides.files}
    matched_objects: set[str] = set()
    matched_files: set[str] = set()
    object_applications: dict[tuple[str, FrameType], int] = defaultdict(int)
    resolved: list[ClassifiedFrame] = []

    for frame in frames:
        filename = frame.header.path.name
        file_rule = file_rules.get(filename)
        if file_rule is not None:
            matched_files.add(filename)
        target_frametype = _header_target_frametype(frame.header)
        if target_frametype is None:
            if file_rule is not None and file_rule.frametype is not None:
                frametype = file_rule.frametype
                resolved.append(
                    replace(
                        frame,
                        kind=_raw_kind_from_frametype(frametype, frame.kind),
                        frametype_override=frametype,
                        frametype_source="file override",
                    )
                )
                review_notes.append(
                    f"{filename}: frametype={frametype.value} applied from file override"
                )
                continue
            if file_rule is not None:
                review_notes.append(
                    f"{filename}: file override was not applied because this is not a target "
                    "or standard exposure"
                )
            resolved.append(frame)
            continue

        header = frame.header
        object_source = frame.object_source
        if file_rule is not None and file_rule.object_name is not None:
            previous = header.object_name
            header = replace(header, object_name=file_rule.object_name)
            object_source = "file override"
            review_notes.append(
                f"{filename}: object={file_rule.object_name!r} applied from file override "
                f"(header OBJECT={previous!r})"
            )

        object_rule = object_rules.get(header.object_name or "")
        if object_rule is not None:
            matched_objects.add(object_rule.object_name)
        if file_rule is not None and file_rule.frametype is not None:
            frametype = file_rule.frametype
            source = "file override"
            review_notes.append(
                f"{filename}: frametype={frametype.value} applied from file override"
            )
        elif object_rule is not None:
            frametype = object_rule.frametype
            source = "object override"
            object_applications[(object_rule.object_name, frametype)] += 1
        else:
            resolved.append(replace(frame, header=header, object_source=object_source))
            continue

        kind = frame.kind
        if frametype is FrameType.SCIENCE:
            kind = RawFrameKind.SCIENCE
        elif frametype is FrameType.STANDARD:
            kind = RawFrameKind.STANDARD
        resolved.append(
            replace(
                frame,
                header=header,
                kind=kind,
                object_source=object_source,
                frametype_override=frametype,
                frametype_source=source,
            )
        )

    for (object_name, frametype), count in sorted(
        object_applications.items(), key=lambda item: (item[0][0].casefold(), item[0][1].value)
    ):
        review_notes.append(
            f"{object_name}: frametype={frametype.value} applied from object override "
            f"to {count} exposure(s)"
        )
    for object_name in sorted(set(object_rules).difference(matched_objects), key=str.casefold):
        review_notes.append(f"{object_name}: object override matched no target exposures")
    for filename in sorted(set(file_rules).difference(matched_files), key=str.casefold):
        review_notes.append(f"{filename}: file override matched no FITS file")
    return tuple(resolved)


def _classification_override_suggestions(
    frames: tuple[ClassifiedFrame, ...],
) -> tuple[tuple[ObjectOverride, ...], tuple[str, ...]]:
    grouped = _target_classifications_by_object(frames, final=False)
    suggestions: list[ObjectOverride] = []
    notes: list[str] = []
    for object_name in sorted(grouped, key=str.casefold):
        classifications = {frametype for _, frametype in grouped[object_name]}
        expected = _heuristic_frametype(object_name)
        if expected is None or classifications == {expected}:
            continue
        rendered = " and ".join(sorted(item.value for item in classifications))
        if len(classifications) > 1:
            detail = f"header classifications include both {rendered}"
        else:
            detail = f"header classification is {rendered}"
        notes.append(f"{object_name}: {detail}; object-name heuristic suggests {expected.value}")
        suggestions.append(ObjectOverride(object_name, expected))
    return tuple(suggestions), tuple(notes)


def _unresolved_classification_notes(
    frames: tuple[ClassifiedFrame, ...], overrides: SetupOverrides
) -> tuple[str, ...]:
    grouped = _target_classifications_by_object(frames, final=True)
    object_rules = {item.object_name for item in overrides.objects}
    file_rules = {item.filename: item for item in overrides.files}
    notes: list[str] = []
    for object_name in sorted(grouped, key=str.casefold):
        entries = grouped[object_name]
        classifications = {frametype for _, frametype in entries}
        expected = _heuristic_frametype(object_name)
        if expected is None or classifications == {expected} or object_name in object_rules:
            continue
        if all(
            (rule := file_rules.get(frame.header.path.name)) is not None
            and rule.frametype is not None
            for frame, _ in entries
        ):
            continue
        rendered = " and ".join(sorted(item.value for item in classifications))
        notes.append(
            f"{object_name}: unresolved classifications include {rendered}; "
            f"object-name heuristic suggests {expected.value}"
        )
    return tuple(notes)


def _target_classifications_by_object(
    frames: tuple[ClassifiedFrame, ...], *, final: bool
) -> dict[str, list[tuple[ClassifiedFrame, FrameType]]]:
    grouped: dict[str, list[tuple[ClassifiedFrame, FrameType]]] = defaultdict(list)
    for frame in frames:
        if not frame.header.object_name or _header_target_frametype(frame.header) is None:
            continue
        frametype = _effective_frametype(frame) if final else _header_target_frametype(frame.header)
        if frametype is not None:
            grouped[frame.header.object_name].append((frame, frametype))
    return grouped


def _effective_frametype(frame: ClassifiedFrame) -> FrameType | None:
    if frame.frametype_override is not None:
        return frame.frametype_override
    if frame.kind is RawFrameKind.SCIENCE:
        return FrameType.SCIENCE
    if frame.kind is RawFrameKind.STANDARD:
        return FrameType.STANDARD
    if frame.kind is RawFrameKind.FLAT:
        return FrameType.FLAT
    if frame.kind is RawFrameKind.ARC_ON:
        return FrameType.ARC_ON
    if frame.kind is RawFrameKind.ARC_OFF:
        return FrameType.ARC_OFF
    if frame.kind is RawFrameKind.DARK:
        return FrameType.DARK
    return None


def _raw_kind_from_frametype(frametype: FrameType, fallback: RawFrameKind) -> RawFrameKind:
    """Translate an explicit file-role override into its setup classification."""

    return {
        FrameType.FLAT: RawFrameKind.FLAT,
        FrameType.ARC_ON: RawFrameKind.ARC_ON,
        FrameType.ARC_OFF: RawFrameKind.ARC_OFF,
        FrameType.DARK: RawFrameKind.DARK,
        FrameType.SCIENCE: RawFrameKind.SCIENCE,
        FrameType.STANDARD: RawFrameKind.STANDARD,
    }.get(frametype, fallback)


def _header_target_frametype(header: RawFrameHeader) -> FrameType | None:
    datatype = _casefold(header.datatype)
    if datatype not in {"target", "standard"}:
        return None
    name_evidence = f"{_casefold(header.object_name)} {header.path.name.casefold()}"
    calibration_evidence = (
        _state_is(header.qth_lamp, "on")
        or _state_is(header.ir_lamp, "on")
        or _state_is(header.arg_lamp, "on")
        or _casefold(header.mode) == "darks"
        or any(_contains_word(name_evidence, token) for token in ("flat", "dark", "arc", "thar"))
    )
    if calibration_evidence:
        return None
    return FrameType.SCIENCE if datatype == "target" else FrameType.STANDARD


def _heuristic_frametype(object_name: str) -> FrameType | None:
    normalized = object_name.strip().casefold()
    if normalized.startswith("hd"):
        return FrameType.STANDARD
    if any(_contains_word(normalized, token) for token in ("standard", "telluric", "std")):
        return None
    return FrameType.SCIENCE


def _filename_object_disagreement_notes(
    frames: tuple[ClassifiedFrame, ...],
) -> tuple[str, ...]:
    disagreements: dict[tuple[str, str], list[str]] = defaultdict(list)
    for frame in frames:
        if _has_expected_calibration_filename_object(frame):
            continue
        header_object = frame.header.object_name
        filename_object = _filename_object_name(frame.header.path.name)
        if (
            header_object
            and filename_object
            and _object_identity_key(header_object) != _object_identity_key(filename_object)
        ):
            disagreements[(header_object, filename_object)].append(frame.header.path.name)

    notes: list[str] = []
    for (header_object, filename_object), filenames in sorted(
        disagreements.items(), key=lambda item: (item[0][0].casefold(), item[0][1].casefold())
    ):
        notes.append(
            f"{header_object}: filename/header object disagreement for {', '.join(filenames)}; "
            f"filename suggests {filename_object!r}, retained FITS OBJECT={header_object!r}"
        )
    return tuple(notes)


def _filename_object_name(filename: str) -> str | None:
    match = re.fullmatch(
        r"[^.]+\.[^.]+\.\d{6}\.(?P<object>[^.]+)\.\d+\.[ab]\.(?:fit|fits|fts)",
        filename,
        flags=re.IGNORECASE,
    )
    return match.group("object") if match else None


def _object_identity_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _has_trusted_dark_signature(header: RawFrameHeader) -> bool:
    """Return whether an iSHELL header has the unambiguous v0.1 dark signature."""

    return (
        _casefold(header.datatype) == "calibration"
        and "dark" in header.path.name.casefold()
        and _casefold(header.slit) == "mirror"
    )


def _has_expected_calibration_filename_object(frame: ClassifiedFrame) -> bool:
    """Return whether a filename/OBJECT difference is normal iSHELL calibration metadata."""

    header = frame.header
    if _has_trusted_dark_signature(header):
        return True
    expected = {
        RawFrameKind.FLAT: ("flat", "qth"),
        RawFrameKind.ARC_ON: ("arc", "thar on"),
        RawFrameKind.ARC_OFF: ("arc", "thar off"),
    }.get(frame.kind)
    if expected is None:
        return False
    filename_marker, expected_object = expected
    return (
        _casefold(header.datatype) == "calibration"
        and filename_marker in header.path.name.casefold()
        and _casefold(header.object_name) == expected_object
    )


def _render_override_suggestions(suggestions: tuple[ObjectOverride, ...]) -> str:
    lines = [
        "# Proposed shuck setup corrections.",
        "# Review, edit, or delete these rules before rerunning shuck setup.",
        "# These suggestions were not applied to the .shuck file generated on this run.",
        "# Raw FITS headers are never modified.",
    ]
    if not suggestions:
        lines.extend(("#", "# No classification corrections were suggested."))
    for suggestion in suggestions:
        lines.extend(
            (
                "",
                f"[objects.{json.dumps(suggestion.object_name, ensure_ascii=False)}]",
                f"frametype = {json.dumps(suggestion.frametype.value)}",
            )
        )
    return "\n".join(lines) + "\n"


def write_control_file(
    control: ControlFile,
    path: str | Path,
    *,
    review_notes: Iterable[str] = (),
) -> None:
    """Serialize a control model as a human-editable format-version-1 file."""

    Path(path).write_text(render_control_file(control, review_notes=review_notes), encoding="utf-8")


def render_control_file(control: ControlFile, *, review_notes: Iterable[str] = ()) -> str:
    """Render a control model using the version-1 INI/table grammar."""

    lines = ["# Generated by shuck setup. Inspect every proposed association before reduction."]
    notes = tuple(review_notes)
    if notes:
        lines.extend(["#", "# REVIEW REQUIRED:"])
        lines.extend(f"# - {note}" for note in notes)
    lines.extend(
        [
            "",
            "[shuck]",
            f"format_version = {control.shuck.format_version}",
            f"raw_path = {control.shuck.raw_path}",
            f"calib_dir = {control.shuck.calib_dir}",
            f"proc_dir = {control.shuck.proc_dir}",
            f"qa_dir = {control.shuck.qa_dir}",
            "",
            "[extraction]",
            f"reduction_mode = {control.extraction.reduction_mode}",
            f"n_apertures = {control.extraction.n_apertures}",
            f"optimal = {_render_bool(control.extraction.optimal)}",
            f"psf_radius = {control.extraction.psf_radius}",
            f"aperture_radius = {control.extraction.aperture_radius}",
            f"background_start = {control.extraction.background_start}",
            f"background_width = {control.extraction.background_width}",
            f"background_degree = {control.extraction.background_degree}",
            f"trace_degree = {control.extraction.trace_degree}",
            "",
            "[combine]",
            f"statistic = {control.combine.statistic}",
            f"sigma_clip = {control.combine.sigma_clip}",
            f"J3_scale_order = {control.combine.j3_scale_order}",
            f"Kgas_scale_order = {control.combine.kgas_scale_order}",
            f"shift_spectra = {_render_bool(control.combine.shift_spectra)}",
            f"prune = {_render_bool(control.combine.prune)}",
            f"shape_correction = {_render_bool(control.combine.shape_correction)}",
            "",
            "[telluric]",
            f"method = {control.telluric.method}",
            f"find_shifts = {_render_bool(control.telluric.find_shifts)}",
            "",
        ]
    )
    _render_table(
        lines,
        "calibrations",
        (
            (
                g.calib_id,
                g.mode,
                _render_file_list(g.flat_files),
                _render_file_list(g.arc_on_files),
                _render_file_list(g.arc_off_files),
            )
            for g in control.calibrations
        ),
    )
    _render_table(
        lines,
        "darks",
        ((g.dark_id, g.itime, g.ndr, g.coadds, _render_file_list(g.files)) for g in control.darks),
    )
    _render_table(
        lines,
        "standards",
        ((s.standard_id, s.target, s.bmag, s.vmag, s.rv_kms) for s in control.standards),
    )
    _render_table(
        lines,
        "data",
        (
            (
                f.filename,
                f.frametype.value,
                f.target or "",
                f.mode or "",
                f.beam or "",
                f.calib or "",
                f.dark or "",
                f.comb_id or "",
                f.telluric_group or "",
            )
            for f in control.data
        ),
    )
    return "\n".join(lines).rstrip() + "\n"


def validate_control_file(control: ControlFile) -> ValidationReport:
    """Run fatal and advisory preflight checks without performing reduction."""

    diagnostics: list[Diagnostic] = []
    error = _diagnostic_adder(diagnostics, Severity.ERROR)
    warning = _diagnostic_adder(diagnostics, Severity.WARNING)
    _validate_v01_settings(control, error)
    calibration_ids = _index_unique(
        ((group.calib_id, group) for group in control.calibrations),
        "calibration ID",
        error,
    )
    dark_ids = _index_unique(((group.dark_id, group) for group in control.darks), "dark ID", error)
    _index_unique(
        ((standard.standard_id, standard) for standard in control.standards),
        "standard ID",
        error,
    )
    raw_directory = _resolve_raw_directory(control)
    header_cache: dict[str, RawFrameHeader | None] = {}

    def checked_header(reference: FileReference, owner: str) -> RawFrameHeader | None:
        if isinstance(reference, Placeholder):
            error("placeholder", f"{owner} contains unresolved placeholder {reference}")
            return None
        if reference in header_cache:
            return header_cache[reference]
        path = raw_directory / reference
        if not path.is_file():
            error("missing-file", f"{owner} references missing file {path}")
            header_cache[reference] = None
            return None
        try:
            header = read_fits_header(path)
        except (OSError, ValueError) as exception:
            error("invalid-fits", f"Could not read {owner} file {path}: {exception}")
            header = None
        header_cache[reference] = header
        return header

    for group in control.calibrations:
        mode = _canonical_mode(group.mode)
        if mode is None:
            error(
                "unsupported-mode",
                f"Calibration {group.calib_id} uses unsupported mode {group.mode!r}",
            )
        if not group.flat_files:
            error("missing-assignment", f"Calibration {group.calib_id} has no flat files")
        if not group.arc_on_files:
            error("missing-assignment", f"Calibration {group.calib_id} has no arc-on files")
        if not group.arc_off_files:
            error("missing-assignment", f"Calibration {group.calib_id} has no arc-off files")
        roles = (
            (RawFrameKind.FLAT, group.flat_files),
            (RawFrameKind.ARC_ON, group.arc_on_files),
            (RawFrameKind.ARC_OFF, group.arc_off_files),
        )
        for role, references in roles:
            for reference in references:
                header = checked_header(reference, f"calibration {group.calib_id} {role.value}")
                if header is None:
                    continue
                if mode is not None and _canonical_mode(header.mode) != mode:
                    error(
                        "incompatible-calibration",
                        f"Calibration {group.calib_id} ({mode}) assigns {header.path.name} "
                        f"with header mode {header.mode!r} as {role.value}",
                    )
                classified_role = classify_raw_frame(header).kind
                if classified_role is not role:
                    error(
                        "incompatible-calibration",
                        f"Calibration {group.calib_id} assigns {header.path.name} as "
                        f"{role.value}, but its headers classify it as {classified_role.value}",
                    )

    for group in control.darks:
        if len(group.files) < SMALL_DARK_GROUP_SIZE:
            warning(
                "small-dark-group",
                f"Dark group {group.dark_id} has only {len(group.files)} frame(s)",
            )
        if not group.files:
            error("missing-assignment", f"Dark group {group.dark_id} has no files")
        for reference in group.files:
            header = checked_header(reference, f"dark group {group.dark_id}")
            if header is None:
                continue
            actual = _detector_configuration(header)
            expected = (group.itime, group.ndr, group.coadds)
            if actual != expected:
                error(
                    "incompatible-dark",
                    f"Dark group {group.dark_id} expects configuration {expected}, but "
                    f"{header.path.name} has {actual}",
                )
            if classify_raw_frame(header).kind is not RawFrameKind.DARK:
                error(
                    "incompatible-dark",
                    f"Dark group {group.dark_id} includes non-dark file {header.path.name}",
                )

    standard_targets = {standard.target: standard for standard in control.standards}
    standard_groups: dict[str, list[DataFrame]] = defaultdict(list)
    for frame in control.data:
        if frame.frametype is FrameType.STANDARD and frame.comb_id:
            standard_groups[frame.comb_id].append(frame)

    for index, frame in enumerate(control.data, start=1):
        owner = f"data row {index} ({frame.filename})"
        header = checked_header(frame.filename, owner)
        if frame.frametype in {
            FrameType.FLAT,
            FrameType.ARC_ON,
            FrameType.ARC_OFF,
            FrameType.DARK,
            FrameType.IGNORE,
        }:
            continue
        mode = _canonical_mode(frame.mode)
        if mode is None:
            error("unsupported-mode", f"{owner} uses unsupported mode {frame.mode!r}")
        beam = _casefold(frame.beam)
        if beam not in {"a", "b"}:
            error("invalid-beam", f"{owner} has invalid beam {frame.beam!r}")
        if beam == "b":
            error(
                "unsupported-beam",
                f"{owner} is active but B-beam frames must be ignored in v0.1",
            )
        if not frame.calib or frame.calib not in calibration_ids:
            error(
                "invalid-reference",
                f"{owner} references unknown calibration ID {frame.calib!r}",
            )
        elif mode is not None and _canonical_mode(calibration_ids[frame.calib].mode) != mode:
            error(
                "incompatible-calibration",
                f"{owner} ({mode}) uses calibration {frame.calib} "
                f"({calibration_ids[frame.calib].mode})",
            )
        if beam == "a":
            if not frame.dark or frame.dark not in dark_ids:
                error(
                    "invalid-reference",
                    f"{owner} has no valid assigned dark (got {frame.dark!r})",
                )
            elif header is not None:
                actual = _detector_configuration(header)
                dark = dark_ids[frame.dark]
                expected = (dark.itime, dark.ndr, dark.coadds)
                if actual != expected:
                    error(
                        "incompatible-dark",
                        f"{owner} has detector configuration {actual}, incompatible with "
                        f"dark {frame.dark} configuration {expected}",
                    )
        if not frame.comb_id:
            error("missing-assignment", f"{owner} has no combination ID")
        if frame.frametype is FrameType.STANDARD:
            if not frame.target or frame.target not in standard_targets:
                error(
                    "invalid-reference",
                    f"{owner} target {frame.target!r} has no standards-table identity",
                )
        elif not frame.telluric_group:
            error("unresolved-telluric", f"{owner} has no assigned telluric group")
        elif frame.telluric_group not in standard_groups:
            error(
                "invalid-reference",
                f"{owner} references unknown telluric group {frame.telluric_group!r}",
            )
        else:
            group_modes = {
                _canonical_mode(standard.mode) for standard in standard_groups[frame.telluric_group]
            }
            if mode is not None and group_modes != {mode}:
                error(
                    "incompatible-telluric",
                    f"{owner} ({mode}) uses telluric group {frame.telluric_group} "
                    f"with modes {sorted(str(value) for value in group_modes)}",
                )
        if header is not None:
            if mode is not None and _canonical_mode(header.mode) != mode:
                error(
                    "header-mismatch",
                    f"{owner} mode {mode} disagrees with FITS header {header.mode!r}",
                )
            if frame.beam and _casefold(header.beam) != beam:
                error(
                    "header-mismatch",
                    f"{owner} beam {frame.beam} disagrees with FITS header {header.beam!r}",
                )

    used_telluric_groups = {
        frame.telluric_group
        for frame in control.data
        if frame.frametype is FrameType.SCIENCE and frame.telluric_group is not None
    }
    referenced_targets = {
        frame.target
        for group_name in used_telluric_groups
        for frame in standard_groups.get(group_name, ())
        if frame.target is not None
    }
    for target in sorted(referenced_targets):
        standard = standard_targets.get(target)
        if standard is None:
            continue
        for field_name in ("bmag", "vmag", "rv_kms"):
            value = getattr(standard, field_name)
            if isinstance(value, Placeholder):
                error(
                    "placeholder",
                    f"Standard {standard.standard_id} ({target}) has unresolved "
                    f"{field_name}={value}",
                )
    return ValidationReport(tuple(diagnostics))


def _split_config_and_tables(
    text: str,
) -> tuple[list[str], dict[str, list[tuple[int, str]]]]:
    config_lines: list[str] = []
    tables: dict[str, list[tuple[int, str]]] = {}
    active_table: str | None = None
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            if active_table is None:
                config_lines.append(raw_line)
            continue
        marker = stripped.casefold().split()
        if len(marker) == 2 and marker[1] == "read":
            name = marker[0]
            if active_table is not None:
                raise ControlFileError(
                    f"Line {line_number}: table {name!r} starts inside {active_table!r}"
                )
            if name not in TABLE_COLUMNS:
                raise ControlFileError(f"Line {line_number}: unknown table {name!r}")
            if name in tables:
                raise ControlFileError(f"Line {line_number}: duplicate table {name!r}")
            active_table = name
            tables[name] = []
            continue
        if len(marker) == 2 and marker[1] == "end":
            name = marker[0]
            if active_table != name:
                raise ControlFileError(
                    f"Line {line_number}: unexpected end marker for table {name!r}"
                )
            active_table = None
            continue
        if active_table is not None:
            tables[active_table].append((line_number, stripped))
        else:
            config_lines.append(raw_line)
    if active_table is not None:
        raise ControlFileError(f"Unterminated table {active_table!r}")
    missing = set(TABLE_COLUMNS).difference(tables)
    if missing:
        raise ControlFileError(f"Missing required table(s): {', '.join(sorted(missing))}")
    return config_lines, tables


def _parse_table(name: str, lines: list[tuple[int, str]]) -> list[dict[str, str]]:
    if not lines:
        raise ControlFileError(f"Table {name!r} is missing its header row")
    header_line, header_text = lines[0]
    header = tuple(cell.strip() for cell in header_text.split("|"))
    expected = TABLE_COLUMNS[name]
    if header != expected:
        raise ControlFileError(
            f"Line {header_line}: table {name!r} columns are {header!r}; expected {expected!r}"
        )
    rows: list[dict[str, str]] = []
    for line_number, line in lines[1:]:
        cells = tuple(cell.strip() for cell in line.split("|"))
        if len(cells) != len(expected):
            raise ControlFileError(
                f"Line {line_number}: table {name!r} has {len(cells)} fields; "
                f"expected {len(expected)}"
            )
        rows.append(dict(zip(expected, cells, strict=True)))
    return rows


def _parse_calibration(row: Mapping[str, str]) -> CalibrationGroup:
    return CalibrationGroup(
        calib_id=_required_cell(row, "calib_id", "calibrations"),
        mode=_required_cell(row, "mode", "calibrations"),
        flat_files=_parse_file_list(row["flat_files"]),
        arc_on_files=_parse_file_list(row["arc_on_files"]),
        arc_off_files=_parse_file_list(row["arc_off_files"]),
    )


def _parse_dark(row: Mapping[str, str]) -> DarkGroup:
    dark_id = _required_cell(row, "dark_id", "darks")
    return DarkGroup(
        dark_id=dark_id,
        itime=_parse_float(row["itime"], f"dark {dark_id} itime"),
        ndr=_parse_int(row["ndr"], f"dark {dark_id} ndr"),
        coadds=_parse_int(row["coadds"], f"dark {dark_id} coadds"),
        files=_parse_file_list(row["files"]),
    )


def _parse_standard(row: Mapping[str, str]) -> Standard:
    standard_id = _required_cell(row, "standard_id", "standards")
    return Standard(
        standard_id=standard_id,
        target=_required_cell(row, "target", "standards"),
        bmag=_parse_numeric_or_placeholder(row["bmag"], f"standard {standard_id} bmag"),
        vmag=_parse_numeric_or_placeholder(row["vmag"], f"standard {standard_id} vmag"),
        rv_kms=_parse_numeric_or_placeholder(row["rv_kms"], f"standard {standard_id} rv_kms"),
    )


def _parse_data_frame(row: Mapping[str, str]) -> DataFrame:
    filename = _parse_reference(_required_cell(row, "filename", "data"))
    try:
        frametype = FrameType(row["frametype"].casefold())
    except ValueError as exception:
        raise ControlFileError(
            f"Data file {filename} has invalid frametype {row['frametype']!r}"
        ) from exception
    return DataFrame(
        filename=filename,
        frametype=frametype,
        target=row["target"] or None,
        mode=row["mode"] or None,
        beam=row["beam"] or None,
        calib=row["calib"] or None,
        dark=row["dark"] or None,
        comb_id=row["comb_id"] or None,
        telluric_group=row["telluric_group"] or None,
    )


def _build_calibration_groups(
    frames: tuple[ClassifiedFrame, ...], review_notes: list[str]
) -> tuple[tuple[CalibrationGroup, ...], dict[str, float]]:
    calibration_kinds = {RawFrameKind.FLAT, RawFrameKind.ARC_ON, RawFrameKind.ARC_OFF}
    candidates: list[tuple[str, dict[RawFrameKind, list[str]], float]] = []
    current: list[tuple[int, ClassifiedFrame]] = []
    use_observation_time = all(frame.header.mjd_obs is not None for frame in frames)

    def finish_current() -> None:
        if not current:
            return
        mode = _canonical_mode(current[0][1].header.mode)
        assert mode is not None
        roles: dict[RawFrameKind, list[str]] = defaultdict(list)
        for _, member in current:
            roles[member.kind].append(member.header.path.name)
        missing = [
            label
            for kind, label in (
                (RawFrameKind.FLAT, "flats"),
                (RawFrameKind.ARC_ON, "arc-on files"),
                (RawFrameKind.ARC_OFF, "arc-off files"),
            )
            if not roles[kind]
        ]
        if missing:
            names = ", ".join(member.header.path.name for _, member in current)
            review_notes.append(
                f"{mode} calibration sequence [{names}] is incomplete "
                f"(missing {', '.join(missing)}) and was not grouped"
            )
        else:
            if use_observation_time:
                sequence_time = sum(
                    member.header.mjd_obs
                    for _, member in current
                    if member.header.mjd_obs is not None
                ) / len(current)
            else:
                sequence_time = (current[0][0] + current[-1][0]) / 2
            candidates.append((mode, roles, sequence_time))
        current.clear()

    for position, frame in enumerate(frames):
        mode = _canonical_mode(frame.header.mode)
        if frame.kind not in calibration_kinds or mode is None:
            finish_current()
            if frame.kind in calibration_kinds:
                review_notes.append(
                    f"{frame.header.path.name}: calibration has unsupported or missing "
                    f"mode {frame.header.mode!r}"
                )
            continue
        if current:
            current_mode = _canonical_mode(current[0][1].header.mode)
            current_has_arcs = any(
                member.kind in {RawFrameKind.ARC_ON, RawFrameKind.ARC_OFF} for _, member in current
            )
            if mode != current_mode or (frame.kind is RawFrameKind.FLAT and current_has_arcs):
                finish_current()
        current.append((position, frame))
    finish_current()

    groups: list[CalibrationGroup] = []
    positions: dict[str, float] = {}
    for number, (mode, roles, position) in enumerate(candidates, start=1):
        calibration_id = f"C{number:02d}"
        groups.append(
            CalibrationGroup(
                calib_id=calibration_id,
                mode=mode,
                flat_files=tuple(roles[RawFrameKind.FLAT]),
                arc_on_files=tuple(roles[RawFrameKind.ARC_ON]),
                arc_off_files=tuple(roles[RawFrameKind.ARC_OFF]),
            )
        )
        positions[calibration_id] = position
    return tuple(groups), positions


def _build_dark_groups(
    frames: tuple[ClassifiedFrame, ...], review_notes: list[str]
) -> tuple[tuple[DarkGroup, ...], dict[tuple[float, int, int], str]]:
    grouped: dict[tuple[float, int, int], list[str]] = defaultdict(list)
    for frame in frames:
        if frame.kind is not RawFrameKind.DARK:
            continue
        configuration = _detector_configuration(frame.header)
        if None in configuration:
            review_notes.append(
                f"{frame.header.path.name}: dark lacks ITIME, NDR, or CO_ADDS and was not grouped"
            )
            continue
        valid_configuration = (configuration[0], configuration[1], configuration[2])
        assert all(value is not None for value in valid_configuration)
        key = (
            float(valid_configuration[0]),
            int(valid_configuration[1]),
            int(valid_configuration[2]),
        )
        grouped[key].append(frame.header.path.name)

    darks: list[DarkGroup] = []
    ids: dict[tuple[float, int, int], str] = {}
    for number, (configuration, files) in enumerate(sorted(grouped.items()), start=1):
        itime, ndr, coadds = configuration
        dark_id = f"D{number:02d}"
        ids[configuration] = dark_id
        darks.append(DarkGroup(dark_id, itime, ndr, coadds, tuple(files)))
    return tuple(darks), ids


def _build_standards(frames: tuple[ClassifiedFrame, ...]) -> tuple[Standard, ...]:
    targets = sorted(
        {
            frame.header.object_name
            for frame in frames
            if _effective_frametype(frame) is FrameType.STANDARD and frame.header.object_name
        },
        key=str.casefold,
    )
    used_ids: set[str] = set()
    standards: list[Standard] = []
    for target in targets:
        standard_id = _unique_identifier(_identifier(target), used_ids)
        used_ids.add(standard_id)
        standards.append(
            Standard(
                standard_id,
                target,
                Placeholder("enter B magnitude"),
                Placeholder("enter V magnitude"),
                Placeholder("enter verified radial velocity in km/s"),
            )
        )
    return tuple(standards)


def _build_data_config(
    header: RawFrameHeader,
    dark_ids: dict[tuple[float, int, int], str],
) -> str | None:
    configuration = _detector_configuration(header)
    if None in configuration:
        return None
    return dark_ids.get((float(configuration[0]), int(configuration[1]), int(configuration[2])))


def _build_data_rows(
    frames: tuple[ClassifiedFrame, ...],
    calibration_groups: tuple[CalibrationGroup, ...],
    calibration_positions: dict[str, float],
    dark_ids: dict[tuple[float, int, int], str],
    review_notes: list[str],
) -> tuple[DataFrame, ...]:
    calibrations_by_mode: dict[str, list[CalibrationGroup]] = defaultdict(list)
    for group in calibration_groups:
        calibrations_by_mode[group.mode].append(group)
    target_ids: dict[str, str] = {}
    used_target_ids: set[str] = set()
    targets = sorted(
        {
            frame.header.object_name
            for frame in frames
            if frame.kind in {RawFrameKind.SCIENCE, RawFrameKind.STANDARD}
            and frame.header.object_name
        },
        key=str.casefold,
    )
    for target in targets:
        identifier = _unique_identifier(_identifier(target), used_target_ids)
        target_ids[target] = identifier
        used_target_ids.add(identifier)
    standard_comb_ids: dict[str, set[str]] = defaultdict(set)
    for frame in frames:
        mode = _canonical_mode(frame.header.mode)
        if (
            _effective_frametype(frame) is FrameType.STANDARD
            and frame.header.object_name
            and _casefold(frame.header.beam) == "a"
            and mode
        ):
            standard_comb_ids[mode].add(
                f"{target_ids[frame.header.object_name]}_{_identifier(mode)}"
            )

    rows: list[DataFrame] = []
    use_observation_time = all(frame.header.mjd_obs is not None for frame in frames)
    for frame_position, frame in enumerate(frames):
        header = frame.header
        mode = _canonical_mode(header.mode) or header.mode
        beam = header.beam.upper() if header.beam else None
        inferred_frametype = _effective_frametype(frame)
        active = inferred_frametype in {FrameType.SCIENCE, FrameType.STANDARD}
        frametype = inferred_frametype if inferred_frametype is not None else FrameType.IGNORE
        if active and beam == "B":
            frametype = FrameType.IGNORE
            review_notes.append(
                f"{header.path.name}: B-beam {inferred_frametype.value} defaults to ignore"
            )
        elif active and beam != "A":
            frametype = FrameType.IGNORE
            review_notes.append(
                f"{header.path.name}: missing or invalid BEAM={header.beam!r}; defaults to ignore"
            )

        target = (
            header.object_name
            if frametype in {FrameType.SCIENCE, FrameType.STANDARD, FrameType.IGNORE}
            else None
        )
        comb_id = calib_id = dark_id = telluric_group = None
        if frametype in {FrameType.SCIENCE, FrameType.STANDARD}:
            if target and mode:
                comb_id = f"{target_ids[target]}_{_identifier(mode)}"
            canonical_mode = _canonical_mode(mode)
            if canonical_mode:
                compatible = calibrations_by_mode[canonical_mode]
                frame_time = header.mjd_obs if use_observation_time else float(frame_position)
                assert frame_time is not None
                distances = {
                    group.calib_id: abs(frame_time - calibration_positions[group.calib_id])
                    for group in compatible
                }
                if distances:
                    nearest_distance = min(distances.values())
                    nearest = [
                        identifier
                        for identifier, distance in distances.items()
                        if distance == nearest_distance
                    ]
                    if len(nearest) == 1:
                        calib_id = nearest[0]
                    else:
                        review_notes.append(
                            f"{header.path.name}: equally near compatible calibrations "
                            f"{', '.join(nearest)}; association left unresolved"
                        )
            dark_id = _build_data_config(header, dark_ids)
            if frametype is FrameType.SCIENCE and canonical_mode:
                possible_groups = standard_comb_ids[canonical_mode]
                if len(possible_groups) == 1:
                    telluric_group = next(iter(possible_groups))
                else:
                    review_notes.append(
                        f"{header.path.name}: telluric association is unresolved; "
                        f"found {len(possible_groups)} standard groups for {canonical_mode}"
                    )
            if calib_id is None:
                review_notes.append(
                    f"{header.path.name}: no unique compatible calibration was proposed"
                )
            if dark_id is None:
                review_notes.append(
                    f"{header.path.name}: no exact ITIME/NDR/CO_ADDS dark match was proposed"
                )
        rows.append(
            DataFrame(
                header.path.name,
                frametype,
                target,
                mode,
                beam,
                calib_id,
                dark_id,
                comb_id,
                telluric_group,
            )
        )
    return tuple(rows)


def _validate_v01_settings(control: ControlFile, error: Callable[[str, str], None]) -> None:
    expected = (
        (control.extraction.reduction_mode, "A-Sky/Dark", "extraction.reduction_mode"),
        (control.extraction.n_apertures, 1, "extraction.n_apertures"),
        (control.extraction.optimal, True, "extraction.optimal"),
        (control.combine.statistic, "robust_weighted_mean", "combine.statistic"),
        (control.combine.j3_scale_order, 402, "combine.J3_scale_order"),
        (control.combine.kgas_scale_order, 224, "combine.Kgas_scale_order"),
        (control.combine.shift_spectra, False, "combine.shift_spectra"),
        (control.combine.prune, False, "combine.prune"),
        (control.combine.shape_correction, False, "combine.shape_correction"),
        (control.telluric.method, "IP", "telluric.method"),
        (control.telluric.find_shifts, False, "telluric.find_shifts"),
    )
    for actual, wanted, name in expected:
        if actual != wanted:
            error("unsupported-setting", f"{name}={actual!r}; v0.1 requires {wanted!r}")


def _diagnostic_adder(
    diagnostics: list[Diagnostic], severity: Severity
) -> Callable[[str, str], None]:
    def add(code: str, message: str) -> None:
        diagnostics.append(Diagnostic(severity, code, message))

    return add


def _index_unique(
    items: Iterable[tuple[str, object]],
    label: str,
    error: Callable[[str, str], None],
) -> dict[str, object]:
    indexed: dict[str, object] = {}
    for identifier, item in items:
        if identifier in indexed:
            error("duplicate-id", f"Duplicate {label} {identifier!r}")
        else:
            indexed[identifier] = item
    return indexed


def _resolve_raw_directory(control: ControlFile) -> Path:
    if control.shuck.raw_path.is_absolute():
        return control.shuck.raw_path
    base = control.source_path.parent if control.source_path is not None else Path.cwd()
    return base / control.shuck.raw_path


def _detector_configuration(
    header: RawFrameHeader,
) -> tuple[float | None, int | None, int | None]:
    return header.itime, header.ndr, header.coadds


def _canonical_mode(value: str | None) -> str | None:
    return {"j3": "J3", "kgas": "Kgas"}.get(_casefold(value))


def _state_is(value: str | None, expected: str) -> bool:
    return _casefold(value) == expected.casefold()


def _casefold(value: str | None) -> str:
    return value.strip().casefold() if value else ""


def _contains_word(value: str, word: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", value) is not None


def _identifier(value: str) -> str:
    identifier = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return identifier or "UNNAMED"


def _unique_identifier(identifier: str, used: set[str]) -> str:
    if identifier not in used:
        return identifier
    number = 2
    while f"{identifier}_{number}" in used:
        number += 1
    return f"{identifier}_{number}"


def _render_table(lines: list[str], name: str, rows: Iterable[Iterable[object]]) -> None:
    rendered_rows = [TABLE_COLUMNS[name]]
    rendered_rows.extend(tuple(str(value) for value in row) for row in rows)
    widths = tuple(
        max(len(row[column]) for row in rendered_rows) for column in range(len(TABLE_COLUMNS[name]))
    )

    lines.append(f"{name} read")
    for row in rendered_rows:
        padded = tuple(value.ljust(width) for value, width in zip(row, widths, strict=True))
        lines.append(" | ".join(padded))
    lines.extend((f"{name} end", ""))


def _render_bool(value: bool) -> str:
    return "true" if value else "false"


def _render_file_list(references: Iterable[FileReference]) -> str:
    return ",".join(str(reference) for reference in references)


def _override_table(document: Mapping[str, object], name: str) -> dict[str, Mapping[str, object]]:
    value = document.get(name, {})
    if not isinstance(value, Mapping):
        raise ControlFileError(f"Override section {name!r} must be a table")
    table: dict[str, Mapping[str, object]] = {}
    for identifier, fields in value.items():
        if not isinstance(identifier, str) or not identifier.strip():
            raise ControlFileError(f"Override section {name!r} has a blank or invalid key")
        if not isinstance(fields, Mapping):
            raise ControlFileError(f"Override {name}.{identifier!r} must be a table")
        table[identifier] = fields
    return table


def _reject_override_fields(values: Mapping[str, object], allowed: set[str], owner: str) -> None:
    unknown = set(values).difference(allowed)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ControlFileError(f"Unknown field(s) for override {owner}: {names}")


def _override_frametype(values: Mapping[str, object], owner: str) -> FrameType:
    value = values.get("frametype")
    if not isinstance(value, str):
        raise ControlFileError(f"Override {owner} must define frametype as a string")
    try:
        return FrameType(value.casefold())
    except ValueError as error:
        choices = ", ".join(item.value for item in FrameType)
        raise ControlFileError(
            f"Override {owner} has invalid frametype {value!r}; expected one of {choices}"
        ) from error


def _require_sections(parser: configparser.ConfigParser, names: Iterable[str]) -> None:
    missing = [name for name in names if not parser.has_section(name)]
    if missing:
        raise ControlFileError(f"Missing required section(s): {', '.join(missing)}")


def _config_string(parser: configparser.ConfigParser, section: str, option: str) -> str:
    try:
        value = parser.get(section, option).strip()
    except (configparser.NoOptionError, configparser.NoSectionError) as exception:
        raise ControlFileError(f"Missing required setting [{section}] {option}") from exception
    if not value:
        raise ControlFileError(f"Setting [{section}] {option} may not be blank")
    return value


def _config_int(parser: configparser.ConfigParser, section: str, option: str) -> int:
    return _parse_int(_config_string(parser, section, option), f"[{section}] {option}")


def _config_float(parser: configparser.ConfigParser, section: str, option: str) -> float:
    return _parse_float(_config_string(parser, section, option), f"[{section}] {option}")


def _config_bool(parser: configparser.ConfigParser, section: str, option: str) -> bool:
    value = _config_string(parser, section, option).casefold()
    if value == "true":
        return True
    if value == "false":
        return False
    raise ControlFileError(f"[{section}] {option} must be true or false, got {value!r}")


def _required_cell(row: Mapping[str, str], name: str, table: str) -> str:
    value = row[name]
    if not value:
        raise ControlFileError(f"Table {table!r} has a blank required field {name!r}")
    return value


def _parse_int(value: str, label: str) -> int:
    try:
        return int(value)
    except ValueError as exception:
        raise ControlFileError(f"{label} must be an integer, got {value!r}") from exception


def _parse_float(value: str, label: str) -> float:
    try:
        return float(value)
    except ValueError as exception:
        raise ControlFileError(f"{label} must be a number, got {value!r}") from exception


def _placeholder(value: str) -> Placeholder | None:
    stripped = value.strip()
    if stripped.startswith("<") and stripped.endswith(">") and len(stripped) > 2:
        return Placeholder(stripped[1:-1])
    return None


def _parse_reference(value: str) -> FileReference:
    return _placeholder(value) or value


def _parse_numeric_or_placeholder(value: str, label: str) -> NumericValue:
    placeholder = _placeholder(value)
    return placeholder if placeholder is not None else _parse_float(value, label)


def _parse_file_list(value: str) -> tuple[FileReference, ...]:
    if not value:
        return ()
    return tuple(_parse_reference(item.strip()) for item in value.split(",") if item.strip())
