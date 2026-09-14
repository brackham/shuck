"""Command-line entry points for shuck."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from shuck.control import (
    ControlFile,
    ControlFileError,
    build_setup_control,
    parse_control_file,
    validate_control_file,
    write_control_file,
)
from shuck.io import write_fits_observation_log
from shuck.pipeline import run_pipeline

STAGES = ("calibrate", "extract", "combine", "telluric", "merge")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shuck")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup = subparsers.add_parser("setup", help="Generate an editable .shuck control file.")
    setup.add_argument(
        "directory",
        metavar="DIRECTORY",
        help="Raw-data directory or night directory containing an immediate raw/ subdirectory.",
    )
    setup.add_argument(
        "-o",
        "--output",
        help="Output path (default: <night-directory>/<night-directory>.shuck).",
    )
    setup.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing control file (required for an existing explicit --output).",
    )
    setup.add_argument(
        "--overrides",
        help="Override TOML path (default: <night-directory>/<night-directory>.overrides.toml).",
    )
    setup.add_argument(
        "--write-log",
        action="store_true",
        help="Write <night-directory>/<night-directory>.obslog.csv from all FITS headers.",
    )

    for stage in STAGES:
        command = subparsers.add_parser(stage, help=f"Run the {stage} stage.")
        command.add_argument("control_file")
        if stage != "calibrate":
            command.add_argument(
                "--reuse-extractions",
                action="store_true",
                help="Reuse current checksummed extraction products instead of recomputing them.",
            )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "setup":
        try:
            result = build_setup_control(
                args.directory,
                output_path=args.output,
                overrides_path=args.overrides,
            )
        except (ControlFileError, OSError, ValueError) as exception:
            print(f"shuck: error: {exception}", file=sys.stderr)
            return 2
        output_path = result.output_path
        reviewed_default_rerun = args.output is None and result.proposed_overrides is None
        if output_path.exists() and not args.overwrite and not reviewed_default_rerun:
            print(f"shuck: error: output file already exists: {output_path}", file=sys.stderr)
            return 2
        try:
            write_control_file(result.control, output_path, review_notes=result.review_notes)
            if result.proposed_overrides is not None:
                with result.overrides_path.open("x", encoding="utf-8") as stream:
                    stream.write(result.proposed_overrides)
            if args.write_log:
                log_path = result.night_directory / f"{result.night_directory.name}.obslog.csv"
                write_fits_observation_log(result.raw_directory, log_path)
        except OSError as exception:
            print(f"shuck: error: {exception}", file=sys.stderr)
            return 2
        print(f"Wrote {output_path}")
        if result.proposed_overrides is not None:
            print(f"Wrote proposed overrides {result.overrides_path}")
        if args.write_log:
            print(f"Wrote {log_path}")
        if result.review_notes:
            print(f"Review required: {len(result.review_notes)} note(s) are recorded in the file.")
        return 0

    control = _preflight_control(
        args.control_file,
        allow_standard_placeholders=args.command in {"calibrate", "extract", "combine"},
    )
    if control is None:
        return 2
    if args.command in STAGES:
        try:
            result = run_pipeline(
                control,
                through=args.command,
                reuse_extractions=getattr(args, "reuse_extractions", False),
            )
        except (OSError, ValueError, RuntimeError) as exception:
            print(f"shuck: error: {exception}", file=sys.stderr)
            return 2
        print(f"Wrote pipeline summary {result.summary_file}")
        return 0
    print(f"shuck: error: command {args.command!r} is not implemented yet", file=sys.stderr)
    return 2


def shuckit_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="shuckit")
    parser.add_argument("control_file")
    parser.add_argument(
        "--reuse-extractions",
        action="store_true",
        help="Reuse current checksummed extraction products instead of recomputing them.",
    )
    args = parser.parse_args(argv)
    control = _preflight_control(args.control_file)
    if control is None:
        return 2
    try:
        result = run_pipeline(
            control,
            through="merge",
            reuse_extractions=args.reuse_extractions,
        )
    except (OSError, ValueError, RuntimeError) as exception:
        print(f"shuck: error: {exception}", file=sys.stderr)
        return 2
    print(f"Wrote pipeline summary {result.summary_file}")
    return 0


def _preflight(path: str | Path) -> bool:
    return _preflight_control(path) is not None


def _preflight_control(
    path: str | Path,
    *,
    allow_standard_placeholders: bool = False,
) -> ControlFile | None:
    try:
        control = parse_control_file(path)
    except ControlFileError as exception:
        print(f"shuck: error: {exception}", file=sys.stderr)
        return None
    report = validate_control_file(control)
    blocking = []
    for diagnostic in report.diagnostics:
        deferred_standard_metadata = (
            allow_standard_placeholders
            and diagnostic.code == "placeholder"
            and diagnostic.message.startswith("Standard ")
        )
        severity = "warning" if deferred_standard_metadata else diagnostic.severity.value
        print(
            f"shuck: {severity}: [{diagnostic.code}] {diagnostic.message}",
            file=sys.stderr,
        )
        if diagnostic.severity.value == "error" and not deferred_standard_metadata:
            blocking.append(diagnostic)
    return control if not blocking else None
