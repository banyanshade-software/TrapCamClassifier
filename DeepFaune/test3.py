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


device = 'cuda'
detector = load_detector('deepfaune-yolov8s_960.pt', device)
classifier = load_classifier('deepfaune-vit_large_patch14_dinov2.lvd142m.v2.pt', device)
exit(0)

# add tracking
# see https://docs.ultralytics.com/modes/track/#persisting-tracks-loop

print("load")
model = YOLO("model/TrapperAI-v02.2024-YOLOv8-m.pt")
print("model loaded")


#vid_path = "/Users/danielbraun/Documents/trackcam/20241109/VD_00003.MP4"
#vid_path = "/Users/danielbraun/Documents/trackcam/20240925/03renard/short.mov"
vid_dir = "/Users/daniel/Documents/trackcam/20260222/100MEDIA/"
#vid_path = f"{vid_dir}/DSCF0064.AVI" # owl
#vid_path = f"{vid_dir}/DSCF0087.AVI" # boar
#vid_path = f"{vid_dir}/DSCF0089.AVI" # deer
#vid_path = f"{vid_dir}/DSCF0069.AVI" # two dears
vid_path = f"{vid_dir}/DSCF0092.AVI" # many boars
#vid_path = f"{vid_dir}/../MOVIE/VD_00042.MP4"
vid_path = f"{vid_dir}/DSCF0009.AVI" # fox
vid_path = f"{vid_dir}/DSCF0005.AVI" # fox
vid_path = f"{vid_dir}/DSCF0027.AVI" # fox

vid_path = "/Users/danielbraun/Documents/trackcam/compil/Chevreuils.mp4"
vid_path = "/Volumes/externe1/trackcam/20260111/DCIM/100MEDIA/DSCF0009.AVI"


cap = cv2.VideoCapture(vid_path)
#backSub = cv2.createBackgroundSubtractorMOG2()
if not cap.isOpened():
    print("Error opening video file")
    exit(1)

# prepare for saving
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
   
size = (frame_width, frame_height) 
print("size: ", size)
SMULT=2
size = (480*SMULT, 270*SMULT)
print("size: ", size)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
vidout = cv2.VideoWriter('/Users/danielbraun/devel/trap/trapperai/tracked.avi',  fourcc, 30, size) 

print("out:", vidout)
#exit(0)

count=0

while cap.isOpened():
    # Capture frame-by-frame
    ret, orgframe = cap.read()
    if ret:
        count += 20
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        #frame = cv2.resize(orgframe, ..)
        #https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
        frame = imutils.resize(orgframe, width=480*SMULT)
        #frame = orgframe
        results = model.track(frame, persist=True)
        print("results n=", len(results)) # how many animals were detected
        annotated_frame = results[0].plot()
        vidout.write(annotated_frame)
        cv2.imshow("YOLO11 Tracking", annotated_frame)
        #cv2.imshow('out', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

print("finished")
cap.release();
vidout.release();
cv2.destroyAllWindows() 
            
#b = results[0].boxes.conf
#print("b=",b)

#c = results[0].boxes.cls # return index value for detection and classification results
#print("c=", c)
