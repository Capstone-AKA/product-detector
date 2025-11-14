from ultralytics import YOLO
import cv2

from config_env import PD, CART_ID, HOST_URL, ADD_ENDPOINT, DEBUG
from product_tracker import ProductTracker
from http_client import HttpPostClient

# 1) Load
model = YOLO(PD)
pt = ProductTracker()
hpc = HttpPostClient(HOST_URL)

# 2) Open and set webcam.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Source unavailable. Check connection or other index(1, 2, ...).")

if DEBUG:
    print('System started...')

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
            vid_stride=3,
            verbose=False,
        )
        
        # 4) Decide whether add product to cart.
        add_list = pt.track_product(results)

        # 5) Send added product list to server.
        if add_list:
            data = {
                'cart_number': CART_ID,
                'product_list': add_list
            }
            hpc.post_json(data, ADD_ENDPOINT)
        
except KeyboardInterrupt:
    pass
finally:
    cap.release()