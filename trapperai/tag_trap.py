#!/usr/bin/env python3
"""
process_files.py — Skeleton for file processing with macOS Finder tagging.

Usage:
    python process_files.py [OPTIONS] FILE [FILE ...]

Options:
    --verbose           Enable verbose output
    --dry-run           Run without applying tags
    --imagefreq FREQ    Example numeric option (default: 1.0)

Dependencies:
    pip install xattr
"""

import argparse
import logging
import plistlib
import sys
from pathlib import Path

import xattr


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
# macOS Finder tagging via xattr
#
# Finder stores tags in the extended attribute com.apple.metadata:_kMDItemUserTags
# as a binary plist containing a list of strings.
# Each entry is either a plain tag name (e.g. "foo") or a color-tagged name
# (e.g. "Green\n2") where the number is the Finder color index:
#   0=none, 1=Gray, 2=Green, 3=Purple, 4=Blue, 5=Yellow, 6=Red, 7=Orange
# ---------------------------------------------------------------------------

MACOS_TAG_XATTR = "com.apple.metadata:_kMDItemUserTags"


def set_finder_tags(filepath: Path, tags: list[str], dry_run: bool = False) -> None:
    """Replace all Finder tags on a file with the given list of tag strings."""
    if dry_run:
        logging.info("[dry-run] Would set tags %s on '%s'", tags, filepath)
        return

    try:
        plist_data = plistlib.dumps(tags, fmt=plistlib.FMT_BINARY)
        xattr.setxattr(str(filepath), MACOS_TAG_XATTR, plist_data)
        logging.debug("Set tags %s on '%s'", tags, filepath)
    except OSError as e:
        logging.error("Failed to set tags on '%s': %s", filepath, e)


def get_finder_tags(filepath: Path) -> list[str]:
    """Return the current Finder tags on a file."""
    try:
        raw = xattr.getxattr(str(filepath), MACOS_TAG_XATTR)
        return plistlib.loads(raw)
    except (OSError, KeyError):
        return []


def add_finder_tag(filepath: Path, tag: str, dry_run: bool = False) -> None:
    """Add a single tag to a file, preserving any existing tags."""
    current = get_finder_tags(filepath)
    # Strip color suffixes (e.g. "Green\n2") before comparing
    existing_names = {t.split("\n")[0] for t in current}
    if tag not in existing_names:
        set_finder_tags(filepath, current + [tag], dry_run=dry_run)
    else:
        logging.debug("Tag '%s' already present on '%s'", tag, filepath)


def clear_finder_tags(filepath: Path, dry_run: bool = False) -> None:
    """Remove all Finder tags from a file."""
    set_finder_tags(filepath, [], dry_run=dry_run)


# ---------------------------------------------------------------------------
# Processing logic  <- YOUR CODE GOES HERE
# ---------------------------------------------------------------------------

def process_file(filepath: Path, args: argparse.Namespace) -> str:
    """
    Analyse a single file and return a tag string.

    The returned string will be applied as a Finder tag on the file.
    Replace the body of this function with your actual logic.
    Use `args` to access any CLI options you need (e.g. args.imagefreq).

    Examples:
        return "reviewed"
        return "low-quality"
        return "Green"   # also works with standard Finder color names
    """
    logging.debug("Processing: %s", filepath)

    # ------------------------------------------------------------------ #
    #  TODO: implement your processing here                                #
    #                                                                      #
    #  Example:                                                            #
    #    data = filepath.read_bytes()                                      #
    #    score = analyse(data, freq=args.imagefreq)                        #
    #    return "high" if score > 0.8 else "low"                           #
    # ------------------------------------------------------------------ #

    return "unprocessed"  # stub


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
        help="Simulate processing without applying any tags",
    )
    parser.add_argument(
        "--imagefreq",
        type=float,
        default=1.0,
        metavar="FREQ",
        help="Example numeric option passed to process_file() (default: 1.0)",
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
            tag = process_file(filepath, args)
        except Exception as exc:  # noqa: BLE001
            logging.error("Error processing '%s': %s", filepath, exc)
            add_finder_tag(filepath, "error", dry_run=args.dry_run)
            fail_count += 1
            continue

        add_finder_tag(filepath, tag, dry_run=args.dry_run)
        logging.info("✓ %s → tag: '%s'", filepath.name, tag)
        ok_count += 1

    logging.info("\nDone — %d ok, %d failed, %d skipped", ok_count, fail_count, skip_count)
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
