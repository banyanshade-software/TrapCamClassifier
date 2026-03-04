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

import torch
import timm
import numpy as np
from torchvision import transforms
from ultralytics import YOLO
from tqdm import tqdm
import cv2

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

    load_model()

    for raw_path in args.files:
        filepath = Path(raw_path)

        if not filepath.exists():
            logging.warning("File not found, skipping: %s", filepath)
            skip_count += 1
            continue

        taglist = process_file(filepath, args)
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


CLASSIFIER_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

CLASS_NAMES = [
    "BADGER", "BEAR", "BEAVER", "BIRD", "BISON",
    "CAT", "CHAMOIS", "COW", "DOG", "EQUID",
    "FALLOW_DEER", "FOX", "GENET", "GOAT", "HEDGEHOG",
    "IBEX", "LAGOMORPH", "LYNX", "MARMOT", "MICROMAMMAL",
    "MOUFLON", "MOOSE", "MUSTELID", "NUTRIA", "OTTER",
    "RACCOON", "RED_DEER", "REINDEER", "ROE_DEER", "SHEEP",
    "SQUIRREL", "WILD_BOAR", "WOLF", "WOLVERINE",
    "HUMAN", "VEHICLE",
]


def load_detector(weights_path: str, device: str) -> YOLO:
    """Load the DeepFaune YOLOv8s detector."""
    print(f"[detector]  loading {weights_path}")
    model = YOLO(weights_path)
    model.to(device)
    return model


def load_classifier(weights_path: str, device: str) -> torch.nn.Module:
    """Load the DeepFaune ViT-L/14 DINOv2 classifier."""
    print(f"[classifier] loading {weights_path}")
    num_classes = len(CLASS_NAMES)
    model = timm.create_model(
        "vit_large_patch14_dinov2.lvd142m",
        pretrained=False,
        num_classes=num_classes,
    )
    ckpt = torch.load(weights_path, map_location=device)
    # DeepFaune checkpoints may be stored under a 'model' key
    state_dict = ckpt.get("model", ckpt)
    # Strip any 'module.' prefix from DataParallel checkpoints
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    return model

def detect_animals(detector: YOLO, frame: np.ndarray, conf_threshold: float = 0.25):
    """
    Run the YOLOv8 detector on a single BGR frame.
    Returns a list of (x1, y1, x2, y2, conf, class_id) tuples.
    """
    results = detector.predict(frame, imgsz=960, conf=conf_threshold, verbose=False)
    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            boxes.append((x1, y1, x2, y2, conf, cls))
    return boxes



@torch.no_grad()
def classify_crop(classifier: torch.nn.Module,
                  crop: np.ndarray,
                  device: str) -> tuple[str, float]:
    """
    Classify a BGR crop with the ViT classifier.
    Returns (predicted_label, confidence_score).
    """
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = CLASSIFIER_TRANSFORM(rgb).unsqueeze(0).to(device)
    logits = classifier(tensor)
    probs  = torch.softmax(logits, dim=1)[0]
    idx    = int(probs.argmax())
    return CLASS_NAMES[idx], float(probs[idx])


# -----------------------------------------

detector = None
classifier = None
device = 'cpu'

def load_model():
    global detector;
    global classifier;
    detector = load_detector('deepfaune-yolov8s_960.pt', device)
    classifier = load_classifier('deepfaune-vit_large_patch14_dinov2.lvd142m.v3.pt', device)
    #print(f"detector {detector}")
    #print(f"classifier {classifier}")

    
def process_file(vid_path: Path, args: argparse.Namespace) -> str:
    #return ['t1', 't2']
    #print(f"detector {detector}")
    """
    Analyse a single file and return a tag string.
    Returns list of tags
    """
    logging.debug("Processing: %s", vid_path)
    #
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        sys.exit(f"[error] Cannot open video: {vid_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"[video] {vid_path}  |  {total_frames} frames  |  {fps:.1f} fps")

    frame_idx = 0
    r = tqdm(total=total_frames, unit="frame", desc="Processing")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only analyse every N-th frame to speed things up
        if frame_idx % 5 == 0:
            timestamp = frame_idx / fps

        boxes = detect_animals(detector, frame, conf_threshold=0.25)
        #print(f" boxes {boxes}")
        if not boxes:
            continue
        for (x1, y1, x2, y2, det_conf, _cls) in boxes:
            h, w = frame.shape[:2]
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            crop = frame[y1c:y2c, x1c:x2c]

            if crop.size == 0:
                continue
            species, cls_conf = classify_crop(classifier, crop, device)
            print(f"idx {frame_idx} specie {species} conf {cls_conf}")


    return []




# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    sys.exit(main())

