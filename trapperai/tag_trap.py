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

import cv2
import imutils
from ultralytics import YOLO
from collections import Counter
from collections import defaultdict

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
            taglist = process_file(filepath, args)
        except Exception as exc:  # noqa: BLE001
            logging.error("Error processing '%s': %s", filepath, exc)
            add_finder_tag(filepath, "error", dry_run=args.dry_run)
            fail_count += 1
            continue

        clear_finder_tags(filepath)
        logging.info("✓ %s → tag: '%s'", filepath.name, taglist)
        for tag in taglist:
            add_finder_tag(filepath, tag, dry_run=args.dry_run)
        ok_count += 1

    logging.info("\nDone — %d ok, %d failed, %d skipped", ok_count, fail_count, skip_count)
    return 0 if fail_count == 0 else 1




# ---------------------------------------------------------------------------
# Processing logic  <- YOUR CODE GOES HERE
# ---------------------------------------------------------------------------

# constants 
MIN_CONFIDENCE = 0.5
MIN_MULTI = 1.5
#

def process_file(vid_path: Path, args: argparse.Namespace) -> str:
    #return ['t1', 't2']
    """
    Analyse a single file and return a tag string.
    Returns list of tags
    """
    logging.debug("Processing: %s", vid_path)
    
    #
    print("load")
    model = YOLO("model/TrapperAI-v02.2024-YOLOv8-m.pt")
    print("model loaded")
    
    
    
    results = model.predict(
    	source=vid_path,
    	#imgsz=320,
    	vid_stride=15,
       	stream=True,
        verbose=False
    	)
    
    
    all_labels = []
    summary = defaultdict(list)
    
    #print(f"results: {len(results)}")
    # only have len without stream=True
    
    for r in results:
        print(f"result----------------------{len(r.boxes)}")
        #r.show()
        n = 0
        for c in r.boxes:
            lbl = model.names[int(c.cls)]
            cf = float(c.conf)
            print(f"c label {lbl} confidence {cf}")
            if cf < MIN_CONFIDENCE:
                continue
            summary[lbl].append(cf)
            n += cf
    
        if n>=MIN_MULTI:
            summary['multi'].append(n)
    
        labels = [model.names[int(c)] for c in r.boxes.cls]
        all_labels.extend(labels)
    
    #xsummary = Counter(all_labels)
    #print(xsummary)
    
    print("---- summary :")
    print(summary)
    
    sum2 = {}
    long = False
    for lbl,lcf in summary.items():
        l = len(lcf)
        print(f"summary .... {lbl} .... {l}")
        if l <= 2:
            continue
        sum2[lbl] = lcf
        if l>7:
            long = True
    
    taglist = list(sum2.keys())
    if len(taglist) >= 3:
        # check for multiple species in same video
        # we obviously have "multi" areay sets
        taglist.append('multiplespecies')
    
    if long == True:
        taglist.append('long')
    
    print(f"tags: {taglist}")
    #print(sum2.keys())
    return taglist





# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    sys.exit(main())

