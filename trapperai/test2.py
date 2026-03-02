import cv2
import imutils
from ultralytics import YOLO

print("load")
model = YOLO("model/TrapperAI-v02.2024-YOLOv8-m.pt")
print("model loaded")


#vid_path = "/Users/danielbraun/Documents/trackcam/20241109/VD_00003.MP4"
#vid_path = "/Users/danielbraun/Documents/trackcam/20240925/03renard/short.mov"
#vid_path = "/Users/danielbraun/Downloads/202412.MP4"
vid_dir = "/Users/daniel/Documents/trackcam/20260222/100MEDIA/"
#vid_path = f"{vid_dir}/DSCF0064.AVI" # owl
#vid_path = f"{vid_dir}/DSCF0087.AVI" # boar
vid_path = f"{vid_dir}/DSCF0089.AVI" # deer


cap = cv2.VideoCapture(vid_path)
#backSub = cv2.createBackgroundSubtractorMOG2()
if not cap.isOpened():
    print("Error opening video file")
    exit(1)

# prepare for saving
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
   
size = (frame_width, frame_height) 
size = (480*2, 270*2)
print("size: ", size)
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#vidout = cv2.VideoWriter('tracked.avi',  fourcc, 30, size) 
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#vidout = cv2.VideoWriter('/Users/danielbraun/devel/trap/trapperai/tracked.mp4',  fourcc, 10, size) 
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
vidout = cv2.VideoWriter('/Users/danielbraun/devel/trap/trapperai/tracked.avi',  fourcc, 30, size) 

print("out:", vidout)
#exit(0)

while cap.isOpened():
    # Capture frame-by-frame
    ret, orgframe = cap.read()
    if ret:
        #frame = cv2.resize(orgframe, ..)
        #https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
        frame = imutils.resize(orgframe, width=480*2)
        #frame = orgframe
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
        vidout.write(frame)
        cv2.imshow('out', frame)
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
