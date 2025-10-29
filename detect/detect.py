from ultralytics import YOLO
from dotenv import load_dotenv

import os

# Load env path.
load_dotenv()
pd = os.getenv('PRODUCT_DETECTOR')
host_url = os.getenv('HOST_URL')
debug = os.getenv('DEBUG')

model = YOLO(pd)

results = model.predict(source=0, imgsz=640, device=0, stream=True, vid_stride=2)

# Access to results.
# for result in results:
#     xywh = result.boxes.xywh
#     xywhn = result.boxes.xywhn  # normalized
#     xyxy = result.boxes.xyxy
#     xyxyn = result.boxes.xyxyn  # normalized
#     names = [result.names[cls.item()] for cls in result.boxes.cls.int()]
#     confs = result.boxes.conf

#     if(debug):
#         print(names)