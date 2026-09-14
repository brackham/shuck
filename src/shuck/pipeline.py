"""Deterministic orchestration for the supported shuck reduction stages."""

from __future__ import annotations

import json
import sys
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
    read_extracted_exposure,
    write_extracted_exposure,
)
from shuck.extraction.preprocess import (
    combine_preprocessed_exposures,
    preprocess_exposure,
)
from shuck.io import read_ishell_raw_metadata
from shuck.merge import merge_orders, write_merged_spectrum
from shuck.qa import (
    write_combination_qa,
    write_extraction_qa,
    write_flat_qa,
    write_merge_qa,
    write_rectification_qa,
    write_telluric_qa,
    write_wavecal_qa,
)
from shuck.resources import spextool_root
from shuck.telluric import (
    TelluricCorrectedSpectrum,
    apply_telluric_correction,
    build_telluric_correction,
    write_corrected_spectrum,
    write_telluric_correction,
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
    telluric_files: tuple[Path, ...]
    merged_files: tuple[Path, ...]
    qa_files: tuple[Path, ...]
    summary_file: Path


def _directory(night: Path, configured: Path) -> Path:
    return configured.resolve() if configured.is_absolute() else (night / configured).resolve()


def _raw_paths(raw_directory: Path, references: tuple[str | Placeholder, ...]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for reference in references:
        if isinstance(reference, Placeholder):
            raise ValueError(f"unresolved input-file placeholder {reference}")
        paths.append(raw_directory / reference)
    return tuple(paths)


def run_pipeline(
    control: ControlFile,
    *,
    through: str = "extract",
    reuse_extractions: bool = False,
) -> PipelineResult:
    """Run calibrated extraction from a validated format-version-1 plan.

    This is the non-interactive orchestration counterpart to the supported
    paths in SpeXTool 5.0.3 ``xspextool.pro`` and
    ``mc_ishellcals2dxd.pro``.
    """

    if through not in {"calibrate", "extract", "combine", "telluric", "merge"}:
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

    spextool = spextool_root()
    detector = load_ishell_detector_calibration(spextool)
    object_frames = tuple(
        frame
        for frame in control.data
        if frame.frametype in {FrameType.SCIENCE, FrameType.STANDARD} and frame.beam == "A"
    )
    frames_by_group = defaultdict(list)
    if through != "calibrate":
        for frame in object_frames:
            if frame.comb_id is None:
                raise ValueError(f"object frame lacks combination group: {frame.filename}")
            frames_by_group[frame.comb_id].append(frame)
    cached_groups: set[str] = set()
    if reuse_extractions:
        for combination_id, frames in frames_by_group.items():
            cache_is_current = True
            for frame in frames:
                if not isinstance(frame.filename, str):
                    cache_is_current = False
                    break
                source = raw_directory / frame.filename
                cached = processing_directory / f"{Path(frame.filename).stem}.extracted.fits"
                if not cached.is_file() or cached.stat().st_mtime < source.stat().st_mtime:
                    cache_is_current = False
                    break
            if cache_is_current:
                cached_groups.add(combination_id)
    frames_to_extract = tuple(
        frame for frame in object_frames if frame.comb_id not in cached_groups
    )
    if through != "calibrate":
        calibration_ids = {frame.calib for frame in frames_to_extract}
        dark_ids = {frame.dark for frame in frames_to_extract}
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
    if through != "calibrate":
        completed = 0
        for combination_id, frames in frames_by_group.items():
            if combination_id in cached_groups:
                print(f"shuck: reusing {len(frames)} cached extractions for {combination_id}")
                for frame in frames:
                    if not isinstance(frame.filename, str) or frame.mode is None:
                        raise ValueError(f"object frame has unresolved metadata: {frame.filename}")
                    source = raw_directory / frame.filename
                    cached = processing_directory / f"{Path(frame.filename).stem}.extracted.fits"
                    plate_scale = load_flat_info(spextool, frame.mode).plate_scale_arcsec_per_pixel
                    extracted_groups[combination_id].append(
                        read_extracted_exposure(
                            cached,
                            source_path=source,
                            metadata=read_ishell_raw_metadata(source),
                            plate_scale_arcsec_per_pixel=plate_scale,
                        )
                    )
                    extracted_files.append(cached)
                    qa_prefix = qa_directory / f"extraction_{Path(frame.filename).stem}"
                    for suffix in (".png", ".json"):
                        qa_path = qa_prefix.with_suffix(suffix)
                        if qa_path.is_file():
                            qa_files.append(qa_path)
                completed += len(frames)
                continue
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
    combined_products = {}
    if through in {"combine", "telluric", "merge"}:
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
            combined_products[combination_id] = combined
            qa_files.extend(
                write_combination_qa(combined, qa_directory / f"combination_{combination_id}")
            )

    telluric_files: list[Path] = []
    corrected_products: dict[str, TelluricCorrectedSpectrum] = {}
    if through in {"telluric", "merge"}:
        standards_by_target = {standard.target: standard for standard in control.standards}
        corrections = {}
        science_combination_ids = tuple(
            dict.fromkeys(
                frame.comb_id
                for frame in object_frames
                if frame.frametype is FrameType.SCIENCE and frame.comb_id is not None
            )
        )
        for combination_id in science_combination_ids:
            science_frames = frames_by_group[combination_id]
            telluric_groups = {frame.telluric_group for frame in science_frames}
            if len(telluric_groups) != 1 or None in telluric_groups:
                raise ValueError(
                    f"science combination {combination_id} lacks one resolved telluric group"
                )
            standard_group = telluric_groups.pop()
            assert standard_group is not None
            if standard_group not in combined_products:
                raise ValueError(f"telluric group {standard_group} has no combined standard")
            standard_frames = frames_by_group[standard_group]
            standard_targets = {frame.target for frame in standard_frames}
            if len(standard_targets) != 1 or None in standard_targets:
                raise ValueError(f"standard group {standard_group} has ambiguous target metadata")
            standard_target = standard_targets.pop()
            assert standard_target is not None
            if standard_target not in standards_by_target:
                raise ValueError(f"no standard metadata exist for target {standard_target}")
            standard_metadata = standards_by_target[standard_target]
            numeric_values = (
                standard_metadata.bmag,
                standard_metadata.vmag,
                standard_metadata.rv_kms,
            )
            if any(isinstance(value, Placeholder) for value in numeric_values):
                raise ValueError(f"standard {standard_target} has unresolved telluric metadata")
            if standard_group not in corrections:
                print(f"shuck: constructing telluric correction {standard_group}")
                correction = build_telluric_correction(
                    combined_products[standard_group],
                    standard_group=standard_group,
                    b_magnitude=float(standard_metadata.bmag),
                    v_magnitude=float(standard_metadata.vmag),
                    radial_velocity_kms=float(standard_metadata.rv_kms),
                    spextool_directory=spextool,
                )
                corrections[standard_group] = correction
                correction_path = write_telluric_correction(
                    correction,
                    processing_directory / f"{standard_group}.telluric.fits",
                )
                telluric_files.append(correction_path)
            correction = corrections[standard_group]
            print(f"shuck: correcting {combination_id} with {standard_group}")
            corrected = apply_telluric_correction(
                combined_products[combination_id],
                correction,
                science_group=combination_id,
            )
            if (
                corrected.science_airmass is not None
                and corrected.standard_airmass is not None
                and abs(corrected.standard_airmass - corrected.science_airmass) > 0.1
            ):
                difference = corrected.standard_airmass - corrected.science_airmass
                print(
                    f"shuck: warning: {combination_id} and {standard_group} differ in "
                    f"airmass by {difference:+.3f}",
                    file=sys.stderr,
                )
            corrected_products[combination_id] = corrected
            corrected_path = write_corrected_spectrum(
                corrected,
                processing_directory / f"{combination_id}.corrected.fits",
            )
            telluric_files.append(corrected_path)
            qa_files.extend(
                write_telluric_qa(
                    correction,
                    corrected,
                    qa_directory / f"telluric_{combination_id}",
                )
            )

    merged_files: list[Path] = []
    if through == "merge":
        for combination_id, corrected in corrected_products.items():
            print(f"shuck: merging {combination_id}")
            merged = merge_orders(
                corrected.orders,
                science_group=corrected.science_group,
                standard_group=corrected.standard_group,
                science_files=corrected.science_files,
                standard_files=corrected.standard_files,
                science_airmass=corrected.science_airmass,
                standard_airmass=corrected.standard_airmass,
                observation_metadata=corrected.science_metadata,
            )
            merged_path = write_merged_spectrum(
                merged,
                processing_directory / f"{combination_id}.merged.fits",
            )
            merged_files.append(merged_path)
            qa_files.extend(write_merge_qa(merged, qa_directory / f"merge_{combination_id}"))

    summary_path = processing_directory / "shuck_pipeline_summary.json"
    summary = {
        "control_file": str(control.source_path.resolve()),
        "through": through,
        "calibration_files": [str(path) for path in calibration_files],
        "dark_files": [str(path) for path in dark_files],
        "extracted_files": [str(path) for path in extracted_files],
        "combined_files": [str(path) for path in combined_files],
        "telluric_files": [str(path) for path in telluric_files],
        "merged_files": [str(path) for path in merged_files],
        "qa_files": [str(path) for path in qa_files],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return PipelineResult(
        calibration_files=tuple(calibration_files),
        dark_files=tuple(dark_files),
        extracted_files=tuple(extracted_files),
        combined_files=tuple(combined_files),
        telluric_files=tuple(telluric_files),
        merged_files=tuple(merged_files),
        qa_files=tuple(qa_files),
        summary_file=summary_path,
    )
