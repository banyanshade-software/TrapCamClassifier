"""
DeepFaune Video Parser
======================
Processes a video file using the DeepFaune two-stage pipeline:
  1. Detection  – YOLOv8s model finds animals/humans/vehicles in each frame
  2. Classification – ViT-L/14 DINOv2 model identifies the species

Requirements:
    pip install torch torchvision timm ultralytics opencv-python tqdm

Model weights (download separately from https://pbil.univ-lyon1.fr/software/download/deepfaune/):
    - deepfaune-yolov8s_960.pt          (detector,   ~22 MB)
    - deepfaune-vit_large_patch14_dinov2.lvd142m.v2.pt  (classifier, ~1.2 GB)

Usage:
    python deepfaune_video.py --video myvideo.mp4 \
                              --detector deepfaune-yolov8s_960.pt \
                              --classifier deepfaune-vit_large_patch14_dinov2.lvd142m.v2.pt \
                              [--output results.csv] \
                              [--threshold 0.8] \
                              [--frame-skip 5] \
                              [--device cpu]
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import torch
import timm
import numpy as np
from torchvision import transforms
from ultralytics import YOLO
from tqdm import tqdm


# ── Species labels (DeepFaune v1.2, 30 taxa + empty/human/vehicle) ──────────
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

# Imagenet-style normalisation used by timm ViT models
CLASSIFIER_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Model loading ─────────────────────────────────────────────────────────────

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


# ── Inference helpers ─────────────────────────────────────────────────────────

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


# ── Main video loop ───────────────────────────────────────────────────────────

def process_video(args):
    device = args.device

    detector   = load_detector(args.detector, device)
    classifier = load_classifier(args.classifier, device)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"[error] Cannot open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"[video] {args.video}  |  {total_frames} frames  |  {fps:.1f} fps")

    results_rows = []  # accumulated detections

    frame_idx = 0
    pbar = tqdm(total=total_frames, unit="frame", desc="Processing")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only analyse every N-th frame to speed things up
        if frame_idx % args.frame_skip == 0:
            timestamp = frame_idx / fps

            boxes = detect_animals(detector, frame, conf_threshold=0.25)

            if not boxes:
                results_rows.append({
                    "frame":      frame_idx,
                    "timestamp_s": f"{timestamp:.2f}",
                    "detection":  "empty",
                    "bbox":       "",
                    "det_conf":   "",
                    "species":    "EMPTY",
                    "cls_conf":   "",
                    "label":      "EMPTY",
                })
            else:
                for (x1, y1, x2, y2, det_conf, _cls) in boxes:
                    # Safety-clip the bounding box to the frame
                    h, w = frame.shape[:2]
                    x1c, y1c = max(0, x1), max(0, y1)
                    x2c, y2c = min(w, x2), min(h, y2)
                    crop = frame[y1c:y2c, x1c:x2c]

                    if crop.size == 0:
                        continue

                    species, cls_conf = classify_crop(classifier, crop, device)

                    # Apply confidence threshold: below it → "UNDEFINED"
                    label = species if cls_conf >= args.threshold else "UNDEFINED"

                    results_rows.append({
                        "frame":       frame_idx,
                        "timestamp_s": f"{timestamp:.2f}",
                        "detection":   "animal",
                        "bbox":        f"{x1},{y1},{x2},{y2}",
                        "det_conf":    f"{det_conf:.3f}",
                        "species":     species,
                        "cls_conf":    f"{cls_conf:.3f}",
                        "label":       label,
                    })

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    # ── Write CSV ─────────────────────────────────────────────────────────────
    out_path = args.output or str(Path(args.video).stem) + "_deepfaune.csv"
    fieldnames = ["frame", "timestamp_s", "detection",
                  "bbox", "det_conf", "species", "cls_conf", "label"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_rows)

    print(f"\n[done] {len(results_rows)} detections written to {out_path}")

    # ── Console summary ───────────────────────────────────────────────────────
    from collections import Counter
    counts = Counter(r["label"] for r in results_rows)
    print("\n── Species summary ──────────────────")
    for species, n in counts.most_common():
        print(f"  {species:<20} {n:>5} frame(s)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Run DeepFaune two-stage pipeline on a video file."
    )
    p.add_argument("--video",      required=True,
                   help="Path to input video file")
    p.add_argument("--detector",   required=True,
                   help="Path to deepfaune-yolov8s_960.pt")
    p.add_argument("--classifier", required=True,
                   help="Path to deepfaune-vit_large_patch14_dinov2.lvd142m.v2.pt")
    p.add_argument("--output",     default=None,
                   help="Output CSV path (default: <video_name>_deepfaune.csv)")
    p.add_argument("--threshold",  type=float, default=0.8,
                   help="Minimum classifier confidence to assign a label (default: 0.8)")
    p.add_argument("--frame-skip", type=int,   default=5,
                   help="Analyse every N-th frame (default: 5, set 1 for every frame)")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Torch device: cuda / mps / cpu  (auto-detected by default)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_video(args)
