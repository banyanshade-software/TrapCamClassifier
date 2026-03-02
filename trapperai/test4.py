import cv2
import imutils
from ultralytics import YOLO
from collections import Counter
from collections import defaultdict


# add tracking
# see https://docs.ultralytics.com/modes/track/#persisting-tracks-loop


#
# constants 
MIN_CONFIDENCE = 0.5
MIN_MULTI = 1.5
#
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
vid_path = f"{vid_dir}/DSCF0009.AVI" # fox

vid_path = f"{vid_dir}/DSCF0005.AVI" # fox

#vid_path = f"{vid_dir}/../MOVIE/VD_00042.MP4"


results = model.predict(
	source=vid_path,
	#imgsz=320,
	#vid_stride=30,
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
    if l>5:
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
