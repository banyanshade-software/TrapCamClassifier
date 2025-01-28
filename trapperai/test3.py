import cv2
import imutils
from ultralytics import YOLO

# add tracking
# see https://docs.ultralytics.com/modes/track/#persisting-tracks-loop

print("load")
model = YOLO("TrapperAI-v02.2024-YOLOv8-m.pt")
print("model loaded")


#vid_path = "/Users/danielbraun/Documents/trackcam/20241109/VD_00003.MP4"
#vid_path = "/Users/danielbraun/Documents/trackcam/20240925/03renard/short.mov"
vid_path = "/Users/danielbraun/Downloads/DSCF0057.AVI"


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
