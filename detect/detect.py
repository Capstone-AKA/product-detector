from ultralytics import YOLO
import cv2
import threading
import time

from config_env import PD, CART_ID, HOST_URL, ADD_ENDPOINT, DEBUG, TIME_STAMP, CAM_WIDTH, CAM_HEIGHT
from product_tracker import ProductTracker
from http_client import HttpPostClient, t_post_json

# 1) Load
model = YOLO(PD)
pt = ProductTracker(iou_threshold=0.6, count_threshold=3, miss_threshold=2, min_area_norm=0.3)
hpc = HttpPostClient(HOST_URL)

# 2) Open and set webcam.
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

if not cap.isOpened():
    raise RuntimeError("Source unavailable. Check connection or other index(1, 2, ...).")

print('System started...')

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Fail to read frame.")
            break

        t1 = time.time()
        # 3) Predict.
        results = model(
            source=frame,
            imgsz=640,
            conf=0.25,
            device=0,
            verbose=False,
        )
        t2 = time.time()
        
        # 4) Decide whether add product to cart.
        add_list = pt.track_product(results)
        t3 = time.time()

        # 5) Send added product list to server.
        if add_list:
            data = {
                'cart_number': CART_ID,
                'product_list': add_list
            }
            http_t = threading.Thread(target=t_post_json, args=(hpc, data, ADD_ENDPOINT))
            http_t.start()
            # hpc.post_json(data, ADD_ENDPOINT)
        t4 = time.time()

        if TIME_STAMP:
            print(f"Timestamp[infer={t2-t1:.4f}s | track={t3-t2:.4f}s | http={t4-t3:.4f}s | total={t4-t1:.4f}s]")
        
except KeyboardInterrupt:
    pass
finally:
    cap.release()