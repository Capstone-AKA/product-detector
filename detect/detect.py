from ultralytics import YOLO
import cv2

from config_env import PD, HOST_URL, DEBUG
from product_tracker import ProductTracker

# 1) Load
model = YOLO(PD)
pt = ProductTracker()

# 2) Open and set webcam.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Source unavailable. Check connection or other index(1, 2, ...).")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Fail to read frame.")
            break

        # 3) Predict.
        results = model.predict(
            source=frame,
            imgsz=640,
            conf=0.25,
            device=0,
            verbose=false,
        )
        
        # 4) Decide whether add product to cart.
        add_list = pt.track_product(results)
        
except KeyboardInterrupt:
    pass
finally:
    cap.release()