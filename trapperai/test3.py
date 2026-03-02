import cv2
import imutils
from ultralytics import YOLO

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
