"""Command-line entry points for shuck."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


STAGES = ("calibrate", "extract", "combine", "telluric", "merge")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shuck")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup = subparsers.add_parser("setup", help="Generate an editable .shuck control file.")
    setup.add_argument("raw_directory")
    setup.add_argument("-o", "--output", required=True)

    for stage in STAGES:
        command = subparsers.add_parser(stage, help=f"Run the {stage} stage.")
        command.add_argument("control_file")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raise NotImplementedError(f"Command {args.command!r} is not implemented yet.")


def shuckit_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="shuckit")
    parser.add_argument("control_file")
    args = parser.parse_args(argv)
    raise NotImplementedError(
        f"Full reduction for {args.control_file!r} is not implemented yet."
    )
