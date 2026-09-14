import csv
from dataclasses import replace
from pathlib import Path

import pytest
from astropy.io import fits

from shuck.cli import main
from shuck.control import (
    FrameType,
    RawFrameKind,
    build_setup_control,
    classify_raw_frame,
    parse_control_file,
    validate_control_file,
    write_control_file,
)
from shuck.io import read_fits_header


def _write_frame(
    directory: Path,
    filename: str,
    *,
    object_name: str,
    datatype: str,
    mode: str = "J3",
    beam: str = "A",
    slit: str = "0.75",
    itime: float = 10.0,
    ndr: int = 4,
    coadds: int = 1,
    qth_lamp: str = "Off",
    arg_lamp: str = "Off",
    extra_header: dict[str, object] | None = None,
    extension_header: dict[str, object] | None = None,
) -> Path:
    mjd_obs = 60000.0 + len(tuple(directory.glob("*.fits"))) / 1000.0
    header = fits.Header(
        {
            "INSTRUME": "iSHELL Spectrograph",
            "OBJECT": object_name,
            "DATATYPE": datatype,
            "XDTILT": mode,
            "BEAM": beam,
            "SLIT": slit,
            "ITIME": itime,
            "NDR": ndr,
            "IA_NDR": ndr,
            "CO_ADDS": coadds,
            "IA_COADD": coadds,
            "MJD_OBS": mjd_obs,
            "QTH_LAMP": qth_lamp,
            "IR_LAMP": "Off",
            "ARG_LAMP": arg_lamp,
            "WINCOV": "Out",
            "CALMIR": "In" if qth_lamp == "On" or arg_lamp == "On" else "Out",
            "ARLMPSTG": "In" if arg_lamp == "On" else "Out",
        }
    )
    if extra_header:
        header.update(extra_header)
    path = directory / filename
    hdus = [fits.PrimaryHDU(header=header)]
    if extension_header is not None:
        hdus.append(fits.ImageHDU(header=fits.Header(extension_header)))
    fits.HDUList(hdus).writeto(path)
    return path


def _raw_directory(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_frame(
        raw,
        "flat1.fits",
        object_name="QTH",
        datatype="calibration",
        qth_lamp="On",
        extra_header={"TCS_OBJ": "Stale Target"},
    )
    _write_frame(
        raw,
        "arc_on.fits",
        object_name="ThAr on",
        datatype="calibration",
        arg_lamp="On",
        extra_header={"TCS_OBJ": "Stale Target"},
    )
    _write_frame(
        raw,
        "arc_off.fits",
        object_name="ThAr off",
        datatype="calibration",
        beam="B",
        extra_header={"TCS_OBJ": "Stale Target"},
    )
    _write_frame(raw, "dark1.fits", object_name="no_name", datatype="calibration", mode="Darks")
    _write_frame(raw, "dark2.fits", object_name="no_name", datatype="calibration", mode="Darks")
    _write_frame(
        raw,
        "dark3.fits",
        object_name="dark",
        datatype="calibration",
        mode="Darks",
        itime=20.0,
        ndr=8,
    )
    for number in range(4, 7):
        _write_frame(
            raw,
            f"dark{number}.fits",
            object_name="no_name",
            datatype="calibration",
            mode="Darks",
        )
    for number in range(7, 10):
        _write_frame(
            raw,
            f"dark{number}.fits",
            object_name="no_name",
            datatype="calibration",
            mode="Darks",
            itime=20.0,
            ndr=8,
        )
    _write_frame(raw, "science_a.fits", object_name="Planet", datatype="target")
    _write_frame(
        raw,
        "science_long_a.fits",
        object_name="Planet",
        datatype="target",
        itime=20.0,
        ndr=8,
    )
    _write_frame(raw, "science_b.fits", object_name="Planet", datatype="target", beam="B")
    _write_frame(raw, "standard_a.fits", object_name="Std Star", datatype="standard")
    _write_frame(raw, "standard_b.fits", object_name="Std Star", datatype="standard", beam="B")
    _write_frame(
        raw,
        "ambiguous.fits",
        object_name="telluric candidate",
        datatype="target",
    )
    return raw


def test_setup_resolves_explicit_raw_and_night_directories_equivalently(
    tmp_path: Path,
) -> None:
    night = tmp_path / "night"
    night.mkdir()
    raw = _raw_directory(night)
    fits.PrimaryHDU().writeto(night / "unrelated.fits")
    output = tmp_path / "night.shuck"

    explicit_result = build_setup_control(raw, output_path=output)
    night_result = build_setup_control(night, output_path=output)

    assert explicit_result == night_result
    assert explicit_result.night_directory == night
    assert explicit_result.raw_directory == raw
    assert explicit_result.control.shuck.raw_path == Path("night/raw")
    assert explicit_result.control.shuck.calib_dir == Path("night/cal")
    assert (night / "cal").is_dir()
    assert (night / "proc").is_dir()
    assert (night / "qa").is_dir()
    assert not (night / "red").exists()


def test_setup_error_identifies_both_checked_directories(tmp_path: Path) -> None:
    empty_night = tmp_path / "empty-night"
    empty_night.mkdir()

    with pytest.raises(FileNotFoundError) as error:
        build_setup_control(empty_night)

    message = str(error.value)
    assert str(empty_night) in message
    assert str(empty_night / "raw") in message
    assert not (empty_night / "raw").exists()


def test_setup_cli_default_output_and_relative_paths_support_spaces(
    tmp_path: Path,
) -> None:
    night = tmp_path / "night with spaces"
    night.mkdir()
    raw = _raw_directory(night)
    expected_output = night / "night with spaces.shuck"

    assert main(["setup", str(night)]) == 0
    assert expected_output.is_file()
    control = parse_control_file(expected_output)
    assert control.shuck.raw_path == Path("raw")
    assert control.shuck.calib_dir == Path("cal")
    assert control.shuck.proc_dir == Path("proc")
    assert control.shuck.qa_dir == Path("qa")
    assert (night / "cal").is_dir()
    assert (night / "proc").is_dir()
    assert (night / "qa").is_dir()

    assert main(["setup", str(raw), "--overwrite"]) == 0
    assert expected_output.is_file()


def test_setup_classifies_groups_and_writes_explicit_associations(tmp_path: Path) -> None:
    raw = _raw_directory(tmp_path)
    output = tmp_path / "night.shuck"

    result = build_setup_control(raw, output_path=output)

    assert len(result.control.calibrations) == 1
    calibration = result.control.calibrations[0]
    assert calibration.mode == "J3"
    assert calibration.flat_files == ("flat1.fits",)
    assert calibration.arc_on_files == ("arc_on.fits",)
    assert calibration.arc_off_files == ("arc_off.fits",)
    assert [(dark.itime, dark.ndr, dark.coadds) for dark in result.control.darks] == [
        (10.0, 4, 1),
        (20.0, 8, 1),
    ]
    assert len(result.control.darks[0].files) == 5
    assert len(result.control.darks[1].files) == 4

    rows = {str(frame.filename): frame for frame in result.control.data}
    assert rows["science_a.fits"].frametype is FrameType.SCIENCE
    assert rows["science_a.fits"].calib == calibration.calib_id
    assert rows["science_a.fits"].dark == result.control.darks[0].dark_id
    assert rows["science_a.fits"].telluric_group == rows["standard_a.fits"].comb_id
    assert rows["science_long_a.fits"].comb_id == rows["science_a.fits"].comb_id
    assert rows["science_long_a.fits"].dark == result.control.darks[1].dark_id
    assert rows["science_b.fits"].frametype is FrameType.IGNORE
    assert rows["science_b.fits"].calib is None
    assert rows["standard_a.fits"].frametype is FrameType.STANDARD
    assert rows["standard_b.fits"].frametype is FrameType.IGNORE
    assert rows["flat1.fits"].frametype is FrameType.FLAT
    assert rows["arc_on.fits"].frametype is FrameType.ARC_ON
    assert rows["arc_off.fits"].frametype is FrameType.ARC_OFF
    assert rows["dark1.fits"].frametype is FrameType.DARK
    assert rows["ambiguous.fits"].frametype is FrameType.IGNORE
    assert any("ambiguous.fits" in note for note in result.review_notes)
    assert not any("Stale Target" in note or "no_name" in note for note in result.review_notes)

    write_control_file(result.control, output, review_notes=result.review_notes)
    reparsed = parse_control_file(output)
    assert len(reparsed.darks) == 2
    rendered = output.read_text(encoding="utf-8")
    assert "# REVIEW REQUIRED:" in rendered
    assert "[telluric]" in rendered


def test_setup_separates_calibration_sequences_and_assigns_nearest(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_frame(
        raw, "flat_a1.fits", object_name="flat", datatype="calibration", mode="Kgas", qth_lamp="On"
    )
    _write_frame(
        raw, "flat_a2.fits", object_name="flat", datatype="calibration", mode="Kgas", qth_lamp="On"
    )
    _write_frame(
        raw, "arc_a_on.fits", object_name="arc", datatype="calibration", mode="Kgas", arg_lamp="On"
    )
    _write_frame(raw, "arc_a_off.fits", object_name="arc", datatype="calibration", mode="Kgas")
    _write_frame(raw, "science_early.fits", object_name="Planet", datatype="target", mode="Kgas")
    _write_frame(
        raw, "flat_b1.fits", object_name="flat", datatype="calibration", mode="Kgas", qth_lamp="On"
    )
    _write_frame(
        raw, "flat_b2.fits", object_name="flat", datatype="calibration", mode="Kgas", qth_lamp="On"
    )
    _write_frame(
        raw, "arc_b_on1.fits", object_name="arc", datatype="calibration", mode="Kgas", arg_lamp="On"
    )
    _write_frame(
        raw, "arc_b_on2.fits", object_name="arc", datatype="calibration", mode="Kgas", arg_lamp="On"
    )
    _write_frame(raw, "arc_b_off.fits", object_name="arc", datatype="calibration", mode="Kgas")
    _write_frame(raw, "science_late.fits", object_name="Planet", datatype="target", mode="Kgas")
    _write_frame(
        raw,
        "flat_incomplete.fits",
        object_name="flat",
        datatype="calibration",
        mode="Kgas",
        qth_lamp="On",
    )

    result = build_setup_control(raw)

    assert len(result.control.calibrations) == 2
    first, second = result.control.calibrations
    assert first.flat_files == ("flat_a1.fits", "flat_a2.fits")
    assert first.arc_on_files == ("arc_a_on.fits",)
    assert second.flat_files == ("flat_b1.fits", "flat_b2.fits")
    assert second.arc_on_files == ("arc_b_on1.fits", "arc_b_on2.fits")
    rows = {str(frame.filename): frame for frame in result.control.data}
    assert rows["science_early.fits"].calib == first.calib_id
    assert rows["science_late.fits"].calib == second.calib_id
    assert any(
        "flat_incomplete.fits" in note and "incomplete" in note for note in result.review_notes
    )


def test_setup_uses_explicit_calibration_roles_and_complete_five_flat_sequence(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    for number in range(1, 6):
        _write_frame(
            raw,
            f"flat{number}.fits",
            object_name="QTH",
            datatype="calibration",
            qth_lamp="On",
            extra_header={"TCS_OBJ": "Previous Target"},
        )
    _write_frame(
        raw,
        "arc_on.fits",
        object_name="ThAr on",
        datatype="calibration",
        arg_lamp="On",
        extra_header={"TCS_OBJ": "Previous Target"},
    )
    _write_frame(
        raw,
        "arc_off.fits",
        object_name="ThAr off",
        datatype="calibration",
        beam="B",
        extra_header={"TCS_OBJ": "Previous Target"},
    )
    _write_frame(raw, "dark.fits", object_name="no_name", datatype="calibration", mode="Darks")
    _write_frame(raw, "science_a.fits", object_name="Target A", datatype="target")
    _write_frame(raw, "science_b.fits", object_name="Target A", datatype="target", beam="B")
    _write_frame(raw, "standard_b.fits", object_name="Standard", datatype="standard", beam="B")

    result = build_setup_control(raw)
    rows = {str(frame.filename): frame for frame in result.control.data}

    assert len(result.control.calibrations) == 1
    calibration = result.control.calibrations[0]
    assert len(calibration.flat_files) == 5
    assert calibration.arc_on_files == ("arc_on.fits",)
    assert calibration.arc_off_files == ("arc_off.fits",)
    assert rows["flat1.fits"].frametype is FrameType.FLAT
    assert rows["arc_on.fits"].frametype is FrameType.ARC_ON
    assert rows["arc_off.fits"].frametype is FrameType.ARC_OFF
    assert rows["dark.fits"].frametype is FrameType.DARK
    assert rows["science_a.fits"].calib == calibration.calib_id
    assert rows["science_b.fits"].frametype is FrameType.IGNORE
    assert rows["standard_b.fits"].frametype is FrameType.IGNORE
    notes = "\n".join(result.review_notes)
    assert "Previous Target" not in notes
    assert "no_name" not in notes


def test_calibration_classification_is_independent_of_target_standard_metadata(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_frame(raw, "target.fits", object_name="Mixed", datatype="target")
    _write_frame(raw, "standard.fits", object_name="Mixed", datatype="standard")
    calibration = _write_frame(
        raw,
        "qth.fits",
        object_name="QTH",
        datatype="calibration",
        qth_lamp="On",
        extra_header={"TCS_OBJ": "Mixed"},
    )

    classified = classify_raw_frame(read_fits_header(calibration))
    result = build_setup_control(raw)

    assert classified.kind is RawFrameKind.FLAT
    assert (
        next(row for row in result.control.data if row.filename == "qth.fits").frametype
        is FrameType.FLAT
    )
    assert not any("Mixed" in note and "qth.fits" in note for note in result.review_notes)


def test_setup_dark_matching_has_no_integration_time_tolerance(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_frame(
        raw,
        "dark.fits",
        object_name="no_name",
        datatype="calibration",
        mode="Darks",
        itime=10.0,
    )
    _write_frame(
        raw,
        "science.fits",
        object_name="Planet",
        datatype="target",
        itime=10.000001,
    )

    result = build_setup_control(raw)

    science = next(frame for frame in result.control.data if frame.filename == "science.fits")
    assert science.dark is None
    assert any("exact ITIME/NDR/CO_ADDS" in note for note in result.review_notes)


def test_trusted_dark_signature_uses_normalized_header_values_without_object_reviews(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_frame(
        raw,
        "icm.2026A021.260406.DARK_99pt72.00001.a.fits",
        object_name="  NO_NAME  ",
        datatype=" CALIBRATION ",
        slit=" MirRoR ",
    )
    _write_frame(
        raw,
        "icm.2026A021.260406.dark_99pt72.00002.a.fits",
        object_name="no_name",
        datatype="calibration",
        slit="mirror",
    )

    result = build_setup_control(raw)
    rows = {str(frame.filename): frame for frame in result.control.data}

    assert all(row.frametype is FrameType.DARK for row in rows.values())
    assert not any("filename/header object disagreement" in note for note in result.review_notes)
    assert not any("no_name" in note.casefold() for note in result.review_notes)


def test_expected_calibration_filename_object_pairs_do_not_create_review_notes(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_frame(
        raw,
        "icm.2026A021.260406.FLAT.00001.a.fits",
        object_name="  qTh  ",
        datatype=" CALIBRATION ",
        qth_lamp="On",
    )
    _write_frame(
        raw,
        "icm.2026A021.260406.ARC.00002.a.fits",
        object_name=" ThAr On ",
        datatype="calibration",
        arg_lamp="On",
    )
    _write_frame(
        raw,
        "icm.2026A021.260406.arc.00003.b.fits",
        object_name="thar off",
        datatype="calibration",
        beam="B",
    )
    _write_frame(
        raw,
        "icm.2026A021.260406.flat_unknown.00004.a.fits",
        object_name="unknown calibration",
        datatype="calibration",
    )

    result = build_setup_control(raw)
    rows = {str(frame.filename): frame for frame in result.control.data}
    mismatch_notes = [
        note for note in result.review_notes if "filename/header object disagreement" in note
    ]

    assert rows["icm.2026A021.260406.FLAT.00001.a.fits"].frametype is FrameType.FLAT
    assert rows["icm.2026A021.260406.ARC.00002.a.fits"].frametype is FrameType.ARC_ON
    assert rows["icm.2026A021.260406.arc.00003.b.fits"].frametype is FrameType.ARC_OFF
    assert len(mismatch_notes) == 1
    assert "flat_unknown.00004.a.fits" in mismatch_notes[0]


def test_calibration_no_name_without_trusted_dark_signature_is_not_dark(tmp_path: Path) -> None:
    path = _write_frame(
        tmp_path,
        "icm.2026A021.260406.dark_99pt72.00001.a.fits",
        object_name="no_name",
        datatype="calibration",
        slit="0.75",
    )

    classified = classify_raw_frame(read_fits_header(path))

    assert classified.kind is RawFrameKind.AMBIGUOUS
    assert any("no unambiguous lamp/dark classification" in reason for reason in classified.reasons)


def test_conflicting_science_and_calibration_headers_remain_ambiguous(
    tmp_path: Path,
) -> None:
    path = _write_frame(
        tmp_path,
        "conflict.fits",
        object_name="science object",
        datatype="target",
        qth_lamp="On",
    )

    classified = classify_raw_frame(read_fits_header(path))

    assert classified.kind is RawFrameKind.AMBIGUOUS
    assert "conflicts" in classified.reasons[0]


def test_conflicting_detector_keyword_aliases_remain_ambiguous(tmp_path: Path) -> None:
    path = _write_frame(
        tmp_path,
        "conflicting_ndr.fits",
        object_name="Planet",
        datatype="target",
    )
    with fits.open(path, mode="update") as hdus:
        hdus[0].header["IA_NDR"] = 5

    classified = classify_raw_frame(read_fits_header(path))

    assert classified.kind is RawFrameKind.AMBIGUOUS
    assert "conflicting NDR keywords" in classified.reasons[0]


def test_conflicting_object_and_tcs_object_remain_ambiguous(tmp_path: Path) -> None:
    path = _write_frame(
        tmp_path,
        "conflicting_object.fits",
        object_name="HeaderTarget",
        datatype="target",
        extra_header={"TCS_OBJ": "DifferentTarget"},
    )

    header = read_fits_header(path)
    classified = classify_raw_frame(header)

    assert header.object_name is None
    assert classified.kind is RawFrameKind.AMBIGUOUS
    assert "conflicting object identity keywords" in classified.reasons[0]


def test_setup_write_log_is_complete_deterministic_and_read_only(tmp_path: Path) -> None:
    night = tmp_path / "night with spaces"
    raw = night / "raw"
    raw.mkdir(parents=True)
    _write_frame(
        raw,
        "z, first.fits",
        object_name='Target, "Quoted"',
        datatype="target",
        extra_header={
            "TCS_OBJ": 'Target, "Quoted"',
            "ONLY_A": "first-only",
            "COMMENT": "not a scalar column",
        },
        extension_header={"EXTKEY": 'extension, "quoted"'},
    )
    _write_frame(
        raw,
        "a_second.fits",
        object_name="Second Target",
        datatype="target",
        extra_header={"TCS_OBJ": "Second Target", "ONLY_B": "second-only"},
    )
    observatory_log = night / "obslog.txt"
    observatory_log.write_text("observatory-owned\n", encoding="utf-8")
    raw_before = {path.name: path.read_bytes() for path in raw.iterdir()}

    assert main(["setup", str(raw), "--write-log"]) == 0

    log_path = night / "night with spaces.obslog.csv"
    first_log = log_path.read_text(encoding="utf-8")
    with log_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        columns = reader.fieldnames
    assert columns is not None
    assert columns[:7] == [
        "filename",
        "OBJECT",
        "TCS_OBJ",
        "DATATYPE",
        "ITIME",
        "XDTILT",
        "BEAM",
    ]
    assert "ONLY_A" in columns
    assert "ONLY_B" in columns
    assert "HDU1.EXTKEY" in columns
    assert "COMMENT" not in columns
    assert columns.index("ONLY_A") < columns.index("HDU1.EXTKEY") < columns.index("ONLY_B")
    assert [row["filename"] for row in rows] == ["z, first.fits", "a_second.fits"]
    assert rows[0]["ONLY_A"] == "first-only"
    assert rows[0]["ONLY_B"] == ""
    assert rows[1]["ONLY_A"] == ""
    assert rows[1]["ONLY_B"] == "second-only"
    assert rows[0]["HDU1.EXTKEY"] == 'extension, "quoted"'
    assert '"z, first.fits"' in first_log
    assert '"Target, ""Quoted"""' in first_log

    assert main(["setup", str(night), "--write-log"]) == 0
    assert log_path.read_text(encoding="utf-8") == first_log
    assert {path.name: path.read_bytes() for path in raw.iterdir()} == raw_before
    assert observatory_log.read_text(encoding="utf-8") == "observatory-owned\n"


def test_setup_does_not_write_log_without_option(tmp_path: Path) -> None:
    night = tmp_path / "260406"
    raw = night / "raw"
    raw.mkdir(parents=True)
    _write_frame(raw, "science.fits", object_name="Planet", datatype="target")

    assert main(["setup", str(night)]) == 0

    assert not (night / "260406.obslog.csv").exists()


def test_first_run_proposes_override_and_rerun_applies_it_without_rewriting(
    tmp_path: Path,
) -> None:
    night = tmp_path / "260406"
    raw = night / "raw"
    raw.mkdir(parents=True)
    _write_frame(raw, "hd.fits", object_name="HD_106965", datatype="target")

    assert main(["setup", str(night)]) == 0

    control_path = night / "260406.shuck"
    override_path = night / "260406.overrides.toml"
    first_control = parse_control_file(control_path)
    override_text = override_path.read_text(encoding="utf-8")
    assert first_control.data[0].frametype is FrameType.SCIENCE
    assert '[objects."HD_106965"]' in override_text
    assert 'frametype = "standard"' in override_text
    assert "not applied" in override_text

    assert main(["setup", str(night)]) == 0

    second_text = control_path.read_text(encoding="utf-8")
    second_control = parse_control_file(control_path)
    assert second_control.data[0].frametype is FrameType.STANDARD
    assert second_control.standards[0].target == "HD_106965"
    assert "applied from object override" in second_text
    assert override_path.read_text(encoding="utf-8") == override_text

    assert main(["setup", str(night)]) == 0
    assert control_path.read_text(encoding="utf-8") == second_text
    assert override_path.read_text(encoding="utf-8") == override_text


def test_override_precedence_and_file_object_override(tmp_path: Path) -> None:
    night = tmp_path / "260406"
    raw = night / "raw"
    raw.mkdir(parents=True)
    for filename in ("one.fits", "two.fits", "three.fits"):
        _write_frame(raw, filename, object_name="Planet", datatype="target")
    overrides = tmp_path / "custom overrides.toml"
    overrides.write_text(
        """
[objects."Planet"]
frametype = "standard"
bmag = 7.63
vmag = 7.54
rv_kms = -8.36

[objects."Renamed"]
frametype = "standard"

[files."one.fits"]
frametype = "ignore"

[files."two.fits"]
object = "Renamed"
""".lstrip(),
        encoding="utf-8",
    )

    result = build_setup_control(night, overrides_path=overrides)
    rows = {str(frame.filename): frame for frame in result.control.data}

    assert rows["one.fits"].frametype is FrameType.IGNORE
    assert rows["one.fits"].target == "Planet"
    assert rows["two.fits"].frametype is FrameType.STANDARD
    assert rows["two.fits"].target == "Renamed"
    assert rows["three.fits"].frametype is FrameType.STANDARD
    assert rows["three.fits"].target == "Planet"
    assert {standard.target for standard in result.control.standards} == {"Planet", "Renamed"}
    planet = next(standard for standard in result.control.standards if standard.target == "Planet")
    assert planet.bmag == 7.63
    assert planet.vmag == 7.54
    assert planet.rv_kms == -8.36
    notes = "\n".join(result.review_notes)
    assert "one.fits: frametype=ignore applied from file override" in notes
    assert "two.fits: object='Renamed' applied from file override" in notes
    assert "Planet: frametype=standard applied from object override" in notes
    assert result.proposed_overrides is None


def test_classification_suggestions_are_object_level_and_skip_calibrations(
    tmp_path: Path,
) -> None:
    night = tmp_path / "260406"
    raw = night / "raw"
    raw.mkdir(parents=True)
    _write_frame(raw, "hd.fits", object_name="HD_106965", datatype="target")
    _write_frame(raw, "wasp.fits", object_name="WASP107", datatype="standard")
    _write_frame(raw, "mixed_science.fits", object_name="Mixed", datatype="target")
    _write_frame(raw, "mixed_standard.fits", object_name="Mixed", datatype="standard")
    _write_frame(
        raw,
        "flat.fits",
        object_name="HD_CALIBRATION",
        datatype="calibration",
        qth_lamp="On",
    )

    result = build_setup_control(night)
    proposals = result.proposed_overrides
    notes = "\n".join(result.review_notes)

    assert proposals is not None
    assert '[objects."HD_106965"]' in proposals
    assert '[objects."WASP107"]' in proposals
    assert '[objects."Mixed"]' in proposals
    assert '[objects."HD_CALIBRATION"]' not in proposals
    assert "HD_106965: header classification is science" in notes
    assert "WASP107: header classification is standard" in notes
    assert "Mixed: header classifications include both science and standard" in notes


def test_filename_object_disagreement_warns_without_renaming(tmp_path: Path) -> None:
    night = tmp_path / "260406"
    raw = night / "raw"
    raw.mkdir(parents=True)
    filename = "icm.2026A021.260406.FilenameTarget.00030.a.fits"
    _write_frame(raw, filename, object_name="HeaderTarget", datatype="target")

    result = build_setup_control(night)

    assert result.control.data[0].target == "HeaderTarget"
    assert any(
        "filename/header object disagreement" in note
        and "retained FITS OBJECT='HeaderTarget'" in note
        for note in result.review_notes
    )


def _completed_setup_control(tmp_path: Path):
    raw = _raw_directory(tmp_path)
    output = tmp_path / "night.shuck"
    control = build_setup_control(raw, output_path=output).control
    completed_standard = replace(control.standards[0], bmag=7.6, vmag=7.5, rv_kms=12.0)
    return replace(control, standards=(completed_standard,))


def test_preflight_accepts_complete_associations_and_warns_for_small_darks(
    tmp_path: Path,
) -> None:
    control = _completed_setup_control(tmp_path)

    report = validate_control_file(control)

    assert report.ok
    dark_warnings = [item for item in report.warnings if item.code == "small-dark-group"]
    assert len(dark_warnings) == 1
    assert control.darks[1].dark_id in dark_warnings[0].message


def test_preflight_reports_missing_file_with_data_filename(tmp_path: Path) -> None:
    control = _completed_setup_control(tmp_path)
    science_index = next(
        index for index, frame in enumerate(control.data) if frame.frametype is FrameType.SCIENCE
    )
    frames = list(control.data)
    frames[science_index] = replace(frames[science_index], filename="missing.fits")

    report = validate_control_file(replace(control, data=tuple(frames)))

    assert not report.ok
    assert any(
        item.code == "missing-file" and "missing.fits" in item.message for item in report.errors
    )


def test_preflight_reports_invalid_and_unresolved_data_associations(tmp_path: Path) -> None:
    control = _completed_setup_control(tmp_path)
    science_index = next(
        index for index, frame in enumerate(control.data) if frame.frametype is FrameType.SCIENCE
    )
    frames = list(control.data)
    frames[science_index] = replace(
        frames[science_index], calib="C99", dark=None, telluric_group=None
    )

    report = validate_control_file(replace(control, data=tuple(frames)))

    messages = "\n".join(item.message for item in report.errors)
    assert "C99" in messages
    assert "assigned dark" in messages
    assert "telluric group" in messages
    assert "science_a.fits" in messages


def test_preflight_reports_unsupported_mode_and_incompatible_calibration(
    tmp_path: Path,
) -> None:
    control = _completed_setup_control(tmp_path)
    science_index = next(
        index for index, frame in enumerate(control.data) if frame.frametype is FrameType.SCIENCE
    )
    frames = list(control.data)
    frames[science_index] = replace(frames[science_index], mode="L3")

    report = validate_control_file(replace(control, data=tuple(frames)))

    assert any(item.code == "unsupported-mode" for item in report.errors)

    frames[science_index] = replace(frames[science_index], mode="Kgas")
    incompatible_report = validate_control_file(replace(control, data=tuple(frames)))
    assert any(item.code == "incompatible-calibration" for item in incompatible_report.errors)
    assert any(item.code == "header-mismatch" for item in incompatible_report.errors)


def test_setup_cli_writes_file_and_reduction_command_runs_preflight(tmp_path: Path, capsys) -> None:
    raw = _raw_directory(tmp_path)
    output = tmp_path / "night.shuck"

    assert main(["setup", str(raw), "-o", str(output)]) == 0
    assert output.is_file()
    assert not (tmp_path / f"{tmp_path.name}.shuck").exists()
    assert parse_control_file(output).shuck.format_version == 1

    assert main(["extract", str(output)]) == 2
    captured = capsys.readouterr()
    assert "Wrote" in captured.out
    assert "shuck: error:" in captured.err


def test_setup_cli_requires_overwrite_flag_to_replace_existing_file(tmp_path: Path, capsys) -> None:
    raw = _raw_directory(tmp_path)
    output = tmp_path / "night.shuck"
    output.write_text("keep me", encoding="utf-8")

    assert main(["setup", str(raw), "-o", str(output)]) == 2
    assert output.read_text(encoding="utf-8") == "keep me"
    assert "output file already exists" in capsys.readouterr().err

    assert main(["setup", str(raw), "-o", str(output), "--overwrite"]) == 0
    assert parse_control_file(output).shuck.format_version == 1
    assert not capsys.readouterr().err
