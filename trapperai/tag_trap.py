#!/usr/bin/env python3
"""
process_files.py — Skeleton for file processing with macOS Finder tagging.

Usage:
    python process_files.py [OPTIONS] FILE [FILE ...]

Options:
    --verbose           Enable verbose output
    --dry-run           Run without applying tags
    --imagefreq FREQ    Example numeric option (default: 1)
    --tag-color COLOR   Finder color tag: Red, Orange, Yellow, Green, Blue, Purple, Gray (default: Green)
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=level,
    )


# ---------------------------------------------------------------------------
# macOS tagging helpers
# ---------------------------------------------------------------------------

# Finder color tag names accepted by the `tag` CLI tool (brew install tag)
VALID_COLORS = {"Red", "Orange", "Yellow", "Green", "Blue", "Purple", "Gray"}


def add_finder_tag(filepath: Path, tag: str, dry_run: bool = False) -> None:
    """Add a Finder color tag to a file using the `tag` CLI tool.

    Install the tool with:  brew install tag
    """
    if tag not in VALID_COLORS:
        logging.warning("Unknown tag color '%s'. Choose from: %s", tag, ", ".join(VALID_COLORS))
        return

    if dry_run:
        logging.info("[dry-run] Would tag '%s' with '%s'", filepath, tag)
        return

    try:
        subprocess.run(
            ["tag", "--add", tag, str(filepath)],
            check=True,
            capture_output=True,
            text=True,
        )
        logging.debug("Tagged '%s' with '%s'", filepath, tag)
    except FileNotFoundError:
        logging.error(
            "The `tag` CLI tool is not installed. Run: brew install tag"
        )
    except subprocess.CalledProcessError as e:
        logging.error("Failed to tag '%s': %s", filepath, e.stderr.strip())


def remove_all_finder_tags(filepath: Path, dry_run: bool = False) -> None:
    """Remove all Finder tags from a file."""
    if dry_run:
        logging.info("[dry-run] Would clear tags from '%s'", filepath)
        return

    try:
        subprocess.run(
            ["tag", "--remove", "*", str(filepath)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        logging.error("The `tag` CLI tool is not installed. Run: brew install tag")
    except subprocess.CalledProcessError as e:
        logging.error("Failed to clear tags from '%s': %s", filepath, e.stderr.strip())


# ---------------------------------------------------------------------------
# Processing logic  ← YOUR CODE GOES HERE
# ---------------------------------------------------------------------------

class ProcessingResult:
    """Holds the outcome of processing a single file."""

    def __init__(self, success: bool, tag: str = "Green", details: str = ""):
        self.success = success
        self.tag = tag          # Finder color tag to apply
        self.details = details  # Human-readable summary


def process_file(filepath: Path, args: argparse.Namespace) -> ProcessingResult:
    """
    Analyse a single file and return a ProcessingResult.

    Replace the body of this function with your actual logic.
    Use `args` to access any CLI options you need (e.g. args.imagefreq).
    """
    logging.debug("Processing: %s", filepath)

    # ------------------------------------------------------------------ #
    #  TODO: implement your processing here                                #
    #                                                                      #
    #  Example skeleton:                                                   #
    #    data = filepath.read_bytes()                                      #
    #    score = analyse(data, freq=args.imagefreq)                        #
    #    if score > 0.8:                                                   #
    #        return ProcessingResult(success=True, tag="Green",            #
    #                                details=f"score={score:.2f}")         #
    #    else:                                                              #
    #        return ProcessingResult(success=False, tag="Red",             #
    #                                details=f"score={score:.2f}")         #
    # ------------------------------------------------------------------ #

    # Default stub: always succeed
    return ProcessingResult(success=True, tag=args.tag_color, details="stub result")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process files and tag them in macOS Finder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "files",
        nargs="+",
        metavar="FILE",
        help="One or more files to process",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose/debug output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate processing without applying tags",
    )
    parser.add_argument(
        "--imagefreq",
        type=float,
        default=1.0,
        metavar="FREQ",
        help="Example numeric option passed to process_file() (default: 1.0)",
    )
    parser.add_argument(
        "--tag-color",
        default="Green",
        choices=VALID_COLORS,
        metavar="COLOR",
        help="Default Finder tag color if processing succeeds (default: Green). "
             f"Choices: {', '.join(sorted(VALID_COLORS))}",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    ok_count = 0
    fail_count = 0
    skip_count = 0

    for raw_path in args.files:
        filepath = Path(raw_path)

        if not filepath.exists():
            logging.warning("File not found, skipping: %s", filepath)
            skip_count += 1
            continue

        try:
            result = process_file(filepath, args)
        except Exception as exc:  # noqa: BLE001
            logging.error("Unexpected error processing '%s': %s", filepath, exc)
            add_finder_tag(filepath, "Red", dry_run=args.dry_run)
            fail_count += 1
            continue

        # Apply tag based on result
        add_finder_tag(filepath, result.tag, dry_run=args.dry_run)

        if result.success:
            logging.info("✓ %s — %s [tag: %s]", filepath.name, result.details, result.tag)
            ok_count += 1
        else:
            logging.info("✗ %s — %s [tag: %s]", filepath.name, result.details, result.tag)
            fail_count += 1

    # Summary
    logging.info(
        "\nDone — %d ok, %d failed, %d skipped",
        ok_count, fail_count, skip_count,
    )

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

