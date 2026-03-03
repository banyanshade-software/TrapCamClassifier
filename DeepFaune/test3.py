import sys
import cv2
import torch
import timm
import numpy as np
from torchvision import transforms
from ultralytics import YOLO
from tqdm import tqdm



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
    transforms.Resize((518, 518)),
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



vid_dir = "/Users/daniel/Documents/trackcam/20260222/100MEDIA/"
vid_path = f"{vid_dir}/DSCF0027.AVI" # fox
vid_path = "/Volumes/externe1/trackcam/20260111/DCIM/100MEDIA/DSCF0009.AVI"
vid_path = "/Users/danielbraun/Documents/trackcam/compil/Chevreuils.mp4"



device = 'cpu'
detector = load_detector('deepfaune-yolov8s_960.pt', device)
classifier = load_classifier('deepfaune-vit_large_patch14_dinov2.lvd142m.v3.pt', device)


threshold=0.8
cap = cv2.VideoCapture(vid_path)
if not cap.isOpened():
    sys.exit(f"[error] Cannot open video: {vid_path}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
print(f"[video] {vid_path}  |  {total_frames} frames  |  {fps:.1f} fps")

results_rows = []  # accumulated detections
frame_idx = 0
pbar = tqdm(total=total_frames, unit="frame", desc="Processing")

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
            if (len(boxes)>1):
                print("***** multi")
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
                label = species if cls_conf >= threshold else "UNDEFINED"

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
print(f"\n[done] {results_rows} detections")




