from ultralytics import YOLO
print("load")
model = YOLO("TrapperAI-v02.2024-YOLOv8-m.pt")
print("model loaded")
results = model.predict("DSCF0267.JPG")

print("results n=", len(results)) # how many animals were detected

for r in results:
    r.show()

b = results[0].boxes.conf
print("b=",b)

c = results[0].boxes.cls # return index value for detection and classification results
print("c=", c)
