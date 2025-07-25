import numpy as np
from http_request import HttpPostClient

def filter_detection(detections, conf_threshold = 0.7, area_threshold = 19200):
    filtered = []
 
    for det in detections[0]:
        x1, y1, x2, y2, conf, class_id = det

        if conf < conf_threshold:
            continue

        width = x2 - x1
        height = y2 - y1
        area = width * height

        if area < area_threshold:
            continue

        filtered.append(det)

    return np.array(filtered)

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

class ProductTracker:
    def __init__(self, labels, iou_threshold=0.6, count_threshold=5):
        self.iou_threshold = iou_threshold
        self.count_threshold = count_threshold
        self.trackers = {}
        self.next_id = 0
        self.labels = labels

    def update(self, detections):
        
        new_trackers = {}

        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            best_iou = 0
            best_id = None

            for tid, tracker in self.trackers.items():
                iou = compute_iou([x1,y1,x2,y2], tracker['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid

            http_request_data = []
            if best_iou > self.iou_threshold:
                tracker = self.trackers[best_id]
                tracker['bbox'] = [x1,y1,x2,y2]
                tracker['count'] += 1
                new_trackers[best_id] = tracker
                
                if tracker['count'] == self.count_threshold:
                    product_name = self.labels[int(class_id)]
                    http_request_data.append(product_name)
                    print(f'{product_name} added to cart.')

            else:
                new_trackers[self.next_id] = {
                    'bbox': [x1,y1,x2,y2],
                    'class_id': int(class_id),
                    'count': 1
                }
                self.next_id += 1

        self.trackers = new_trackers

        if len(self.trackers) > 0:
            log='tracker list: '
            for tid, tracker in self.trackers.items():
                log += f"{self.labels[tracker['class_id']]}({tracker['count']}), "
            print(log)
