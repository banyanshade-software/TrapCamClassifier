import cv2
import imutils
from ultralytics import YOLO

print("load")
model = YOLO("TrapperAI-v02.2024-YOLOv8-m.pt")
print("model loaded")


vid_path = "/Users/danielbraun/Documents/trackcam/20241109/VD_00003.MP4"

cap = cv2.VideoCapture(vid_path)
#backSub = cv2.createBackgroundSubtractorMOG2()
if not cap.isOpened():
    print("Error opening video file")
    exit(1)


while cap.isOpened():
    # Capture frame-by-frame
    ret, orgframe = cap.read()
    if ret:
        #frame = cv2.resize(orgframe, ..)
        #https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
        frame = imutils.resize(orgframe, width=512)
        results = model.predict(frame)
        print("results n=", len(results)) # how many animals were detected
        for r in results:
            b = r.boxes 
            for box in b:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]  # Confidence score
                cls = box.cls[0]  # Class ID
                label = f"{model.names[int(cls)]}: {conf:.2f}"  # Add label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, (0, 255, 0), 2) 
        cv2.imshow('out', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


            
#b = results[0].boxes.conf
#print("b=",b)

#c = results[0].boxes.cls # return index value for detection and classification results
#print("c=", c)
