from dataclasses import replace
from pathlib import Path

import pytest

from shuck.control import (
    ControlFileError,
    FrameType,
    Placeholder,
    parse_control_file,
    parse_control_text,
    render_control_file,
)

EXAMPLE = Path(__file__).parents[2] / "examples" / "260406.shuck"
TABLE_NAMES = ("calibrations", "darks", "standards", "data")


def _rendered_table_rows(text: str, name: str) -> list[str]:
    lines = text.splitlines()
    start = lines.index(f"{name} read") + 1
    end = lines.index(f"{name} end")
    return lines[start:end]


def _pipe_positions(line: str) -> list[int]:
    return [index for index, character in enumerate(line) if character == "|"]


def _expected_pipe_positions(rows: list[str]) -> list[int]:
    cells = [[cell.strip() for cell in row.split("|")] for row in rows]
    widths = [max(len(row[column]) for row in cells) for column in range(len(cells[0]))]
    positions: list[int] = []
    offset = 0
    for width in widths[:-1]:
        positions.append(offset + width + 1)
        offset += width + 3
    return positions


def _padded_cells(row: str) -> list[str]:
    cells = row.split("|")
    return [
        cell[(1 if column else 0) : (-1 if column < len(cells) - 1 else None)]
        for column, cell in enumerate(cells)
    ]


def test_example_control_file_parses_with_explicit_placeholders() -> None:
    control = parse_control_file(EXAMPLE)

    assert control.shuck.format_version == 1
    assert [(group.calib_id, group.mode) for group in control.calibrations] == [
        ("C01", "J3"),
        ("C02", "Kgas"),
        ("C03", "Kgas"),
    ]
    assert control.calibrations[0].arc_on_files == ("icm.2026A021.260406.arc.00006.a.fits",)
    assert len(control.darks) == 12
    assert len(control.data) == 220
    assert control.data[0].frametype is FrameType.FLAT
    assert control.data[5].frametype is FrameType.ARC_ON
    assert control.data[6].frametype is FrameType.ARC_OFF
    assert control.data[7].frametype is FrameType.SCIENCE
    assert isinstance(control.standards[0].rv_kms, Placeholder)


def test_rendered_control_file_round_trips() -> None:
    original = parse_control_file(EXAMPLE)
    reparsed = parse_control_text(render_control_file(original))

    assert replace(original, source_path=None) == reparsed


def test_rendered_tables_use_content_widths_and_aligned_separators() -> None:
    rendered = render_control_file(parse_control_file(EXAMPLE))

    for name in TABLE_NAMES:
        rows = _rendered_table_rows(rendered, name)
        expected_positions = _expected_pipe_positions(rows)
        widths = [len(cell) for cell in _padded_cells(rows[0])]
        for row in rows:
            assert _pipe_positions(row) == expected_positions
            assert all(row[position - 1 : position + 2] == " | " for position in expected_positions)
            for cell, width in zip(_padded_cells(row), widths, strict=True):
                assert len(cell) == width
                assert cell == cell.strip().ljust(width)


def test_long_table_value_expands_column_without_truncation() -> None:
    control = parse_control_file(EXAMPLE)
    long_filename = "an_intentionally_much_longer_science_filename_than_any_header.fits"
    data = (replace(control.data[0], filename=long_filename), *control.data[1:])
    modified = replace(control, data=data, source_path=None)

    rendered = render_control_file(modified)
    rows = _rendered_table_rows(rendered, "data")

    assert long_filename in rendered
    assert _pipe_positions(rows[0])[0] == len(long_filename) + 1
    assert parse_control_text(rendered) == modified


def test_rendered_control_file_is_deterministic() -> None:
    control = parse_control_file(EXAMPLE)
    first_render = render_control_file(control)

    assert render_control_file(parse_control_text(first_render)) == first_render


def test_parser_rejects_wrong_table_columns_with_line_number() -> None:
    lines = render_control_file(parse_control_file(EXAMPLE)).splitlines()
    header_index = lines.index("darks read") + 1
    lines[header_index] = "dark_id | itime | files"
    text_value = "\n".join(lines)

    with pytest.raises(ControlFileError, match=r"Line \d+.*darks.*columns"):
        parse_control_text(text_value)


def test_parser_rejects_unknown_format_version() -> None:
    text_value = EXAMPLE.read_text(encoding="utf-8").replace(
        "format_version = 1", "format_version = 2"
    )

    with pytest.raises(ControlFileError, match="format_version=2"):
        parse_control_text(text_value)


def test_parser_uses_fixed_v01_telluric_defaults_when_section_is_absent() -> None:
    text_value = EXAMPLE.read_text(encoding="utf-8").replace(
        "[telluric]\nmethod = IP\nfind_shifts = false\n\n", ""
    )

    control = parse_control_text(text_value)

    assert control.telluric.method == "IP"
    assert not control.telluric.find_shifts
