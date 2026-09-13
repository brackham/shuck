"""Deterministic orchestration for the supported shuck reduction stages."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from shuck.calibration.dark import MasterDark, build_master_dark, write_master_dark
from shuck.calibration.flat import (
    NormalizedFlat,
    build_normalized_flat,
    load_flat_info,
    write_normalized_flat,
)
from shuck.calibration.rectify import (
    DistortionSolution,
    build_distortion_solution,
    write_distortion_solution,
)
from shuck.calibration.wavecal import (
    WavelengthSolution,
    build_wavelength_solution,
    load_wavecal_info,
    write_wavelength_solution,
)
from shuck.combine import combine_exposures, write_combined_spectrum
from shuck.control import ControlFile, FrameType, Placeholder
from shuck.detector import load_ishell_detector_calibration
from shuck.extraction.optimal import (
    ExtractedExposure,
    derive_extraction_model,
    extract_exposure,
    write_extracted_exposure,
)
from shuck.extraction.preprocess import (
    combine_preprocessed_exposures,
    preprocess_exposure,
)
from shuck.qa import (
    write_combination_qa,
    write_extraction_qa,
    write_flat_qa,
    write_rectification_qa,
    write_wavecal_qa,
)


@dataclass(frozen=True)
class CalibrationProducts:
    """In-memory products for one control-file calibration group."""

    flat: NormalizedFlat
    wavelength: WavelengthSolution
    distortion: DistortionSolution


@dataclass(frozen=True)
class PipelineResult:
    """Paths produced by one orchestrated invocation."""

    calibration_files: tuple[Path, ...]
    dark_files: tuple[Path, ...]
    extracted_files: tuple[Path, ...]
    combined_files: tuple[Path, ...]
    qa_files: tuple[Path, ...]
    summary_file: Path


def _directory(night: Path, configured: Path) -> Path:
    return configured if configured.is_absolute() else night / configured


def _raw_paths(raw_directory: Path, references: tuple[str | Placeholder, ...]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for reference in references:
        if isinstance(reference, Placeholder):
            raise ValueError(f"unresolved input-file placeholder {reference}")
        paths.append(raw_directory / reference)
    return tuple(paths)


def _spextool_directory() -> Path:
    value = os.environ.get("SPEXTOOL5_DIR")
    if not value:
        raise ValueError(
            "SPEXTOOL5_DIR must point to the local SpeXTool 5.0.3 root containing "
            "instruments/ishell/data"
        )
    directory = Path(value).expanduser().resolve()
    if not (directory / "instruments" / "ishell" / "data").is_dir():
        raise ValueError(f"SPEXTOOL5_DIR is not a SpeXTool iSHELL tree: {directory}")
    return directory


def run_pipeline(control: ControlFile, *, through: str = "extract") -> PipelineResult:
    """Run calibrated extraction from a validated format-version-1 plan.

    This is the non-interactive orchestration counterpart to the supported
    paths in SpeXTool 5.0.3 ``xspextool.pro`` and
    ``mc_ishellcals2dxd.pro``.
    """

    if through not in {"calibrate", "extract", "combine"}:
        raise ValueError(f"unsupported implemented pipeline endpoint {through!r}")
    if control.source_path is None:
        raise ValueError("pipeline execution requires a control file with a source path")
    night = control.source_path.resolve().parent
    raw_directory = _directory(night, control.shuck.raw_path)
    calibration_directory = _directory(night, control.shuck.calib_dir)
    processing_directory = _directory(night, control.shuck.proc_dir)
    qa_directory = _directory(night, control.shuck.qa_dir)
    for directory in (calibration_directory, processing_directory, qa_directory):
        directory.mkdir(parents=True, exist_ok=True)

    spextool = _spextool_directory()
    detector = load_ishell_detector_calibration(spextool)
    object_frames = tuple(
        frame
        for frame in control.data
        if frame.frametype in {FrameType.SCIENCE, FrameType.STANDARD} and frame.beam == "A"
    )
    if through in {"extract", "combine"}:
        calibration_ids = {frame.calib for frame in object_frames}
        dark_ids = {frame.dark for frame in object_frames}
    else:
        calibration_ids = {group.calib_id for group in control.calibrations}
        dark_ids = {group.dark_id for group in control.darks}

    calibration_products: dict[str, CalibrationProducts] = {}
    calibration_files: list[Path] = []
    qa_files: list[Path] = []
    for group in control.calibrations:
        if group.calib_id not in calibration_ids:
            continue
        print(f"shuck: calibrating {group.calib_id} ({group.mode})")
        flat_info = load_flat_info(spextool, group.mode)
        flat = build_normalized_flat(
            _raw_paths(raw_directory, group.flat_files), detector, flat_info
        )
        flat_path = write_normalized_flat(
            flat,
            calibration_directory / f"normalized_flat_{group.calib_id}.fits",
            calib_id=group.calib_id,
        )
        qa_files.extend(write_flat_qa(flat, qa_directory / f"flat_{group.calib_id}"))
        wave_info = load_wavecal_info(spextool, group.mode)
        wavelength = build_wavelength_solution(
            _raw_paths(raw_directory, group.arc_on_files),
            _raw_paths(raw_directory, group.arc_off_files),
            detector,
            flat,
            wave_info,
        )
        wave_path = write_wavelength_solution(
            wavelength,
            calibration_directory / f"wavecal_{group.calib_id}.fits",
            calib_id=group.calib_id,
        )
        qa_files.extend(write_wavecal_qa(wavelength, qa_directory / f"wavecal_{group.calib_id}"))
        distortion = build_distortion_solution(wavelength, flat, wave_info)
        distortion_path = write_distortion_solution(
            distortion,
            calibration_directory / f"distortion_{group.calib_id}.fits",
            calib_id=group.calib_id,
            mode=group.mode,
        )
        qa_files.extend(
            write_rectification_qa(
                distortion,
                wavelength,
                qa_directory / f"rectification_{group.calib_id}",
            )
        )
        calibration_products[group.calib_id] = CalibrationProducts(
            flat=flat,
            wavelength=wavelength,
            distortion=distortion,
        )
        calibration_files.extend((flat_path, wave_path, distortion_path))

    master_darks: dict[str, MasterDark] = {}
    dark_files: list[Path] = []
    for group in control.darks:
        if group.dark_id not in dark_ids:
            continue
        print(f"shuck: building master dark {group.dark_id}")
        dark = build_master_dark(_raw_paths(raw_directory, group.files), detector)
        path = write_master_dark(
            dark,
            calibration_directory / f"master_dark_{group.dark_id}.fits",
            dark_id=group.dark_id,
        )
        master_darks[group.dark_id] = dark
        dark_files.append(path)

    extracted_files: list[Path] = []
    extracted_groups: dict[str, list[ExtractedExposure]] = defaultdict(list)
    if through in {"extract", "combine"}:
        frames_by_group = defaultdict(list)
        for frame in object_frames:
            if frame.comb_id is None:
                raise ValueError(f"object frame lacks combination group: {frame.filename}")
            frames_by_group[frame.comb_id].append(frame)
        completed = 0
        for combination_id, frames in frames_by_group.items():
            preprocessed_group = []
            for frame in frames:
                if not isinstance(frame.filename, str) or frame.calib is None or frame.dark is None:
                    raise ValueError(f"object frame has unresolved dependencies: {frame.filename}")
                print(
                    f"shuck: preprocessing {completed + len(preprocessed_group) + 1}/"
                    f"{len(object_frames)} {frame.filename}"
                )
                preprocessed_group.append(
                    preprocess_exposure(
                        raw_directory / frame.filename,
                        detector,
                        master_darks[frame.dark],
                        calibration_products[frame.calib].flat,
                        calibration_products[frame.calib].distortion,
                    )
                )
            group_stack = combine_preprocessed_exposures(tuple(preprocessed_group))
            model = derive_extraction_model(
                group_stack,
                trace_degree=control.extraction.trace_degree,
            )
            for frame, preprocessed in zip(frames, preprocessed_group, strict=True):
                assert isinstance(frame.filename, str)
                print(f"shuck: extracting {frame.filename} with {combination_id} group trace")
                extracted = extract_exposure(
                    preprocessed,
                    model=model,
                    trace_degree=control.extraction.trace_degree,
                    psf_radius=control.extraction.psf_radius,
                    aperture_radius=control.extraction.aperture_radius,
                    background_start=control.extraction.background_start,
                    background_width=control.extraction.background_width,
                )
                stem = Path(frame.filename).name.removesuffix(".fits")
                output = write_extracted_exposure(
                    extracted,
                    processing_directory / f"{stem}.extracted.fits",
                )
                extracted_files.append(output)
                extracted_groups[combination_id].append(extracted)
                qa_files.extend(write_extraction_qa(extracted, qa_directory / f"extraction_{stem}"))
            completed += len(frames)

    combined_files: list[Path] = []
    if through == "combine":
        for combination_id, exposures in extracted_groups.items():
            modes = {frame.mode for frame in object_frames if frame.comb_id == combination_id}
            if len(modes) != 1:
                raise ValueError(f"combination group {combination_id} has modes {modes}")
            mode = modes.pop()
            scale_order = (
                control.combine.j3_scale_order if mode == "J3" else control.combine.kgas_scale_order
            )
            print(f"shuck: combining {combination_id} ({len(exposures)} spectra)")
            combined = combine_exposures(
                tuple(exposures),
                scale_order=scale_order,
                sigma_clip=control.combine.sigma_clip,
            )
            combined_path = write_combined_spectrum(
                combined,
                processing_directory / f"{combination_id}.combined.fits",
                combination_id=combination_id,
            )
            combined_files.append(combined_path)
            qa_files.extend(
                write_combination_qa(combined, qa_directory / f"combination_{combination_id}")
            )

    summary_path = processing_directory / "shuck_pipeline_summary.json"
    summary = {
        "control_file": str(control.source_path.resolve()),
        "through": through,
        "calibration_files": [str(path) for path in calibration_files],
        "dark_files": [str(path) for path in dark_files],
        "extracted_files": [str(path) for path in extracted_files],
        "combined_files": [str(path) for path in combined_files],
        "qa_files": [str(path) for path in qa_files],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return PipelineResult(
        calibration_files=tuple(calibration_files),
        dark_files=tuple(dark_files),
        extracted_files=tuple(extracted_files),
        combined_files=tuple(combined_files),
        qa_files=tuple(qa_files),
        summary_file=summary_path,
    )
