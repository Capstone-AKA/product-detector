import numpy as np
from config_env import DEBUG

class Track:
    __slots__ = ("name", "location", "count", "update_flag")
    
    def __init__(self, name, location, count=1):
        self.name = name
        self.location = location
        self.count = count
        self.update_flag = True

    def reset_update_flag(self):
        self.update_flag = False

    def update(self, location):
        self.location = location
        self.count += 1
        self.update_flag = True


class ProductTracker:
    def __init__(self, iou_threshold=0.6, count_threshold=5, min_area_norm=0.4):
        self.iou_threshold = float(iou_threshold)
        self.count_threshold = int(count_threshold)
        self.min_area_norm = float(min_area_norm)
        self.track_list = []    # {'name', 'location', 'count'}

    def _compute_iou(self, boxA, boxB):
        # box: [x1,y1,x2,y2]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interW = max(0.0, xB - xA)
        interH = max(0.0, yB - yA)
        interArea = interW * interH
        areaA = max(0.0, (boxA[2] - boxA[0])) * max(0.0, (boxA[3] - boxA[1]))
        areaB = max(0.0, (boxB[2] - boxB[0])) * max(0.0, (boxB[3] - boxB[1]))
        denom = areaA + areaB - interArea + 1e-6
        return interArea / denom

    def _filter_detection(self, detections):
        """
        detections: Ultralytics results list (len=1 per frame)
        return: [{'xyxy': [x1,y1,x2,y2], 'name': 'class_name'}, ...]
        """
    
        filtered = []
        if not detections:
            return filtered

        det = detections[0]
        if det.boxes is None or len(det.boxes) == 0:
            return filtered

        # Convert tensor to CPU numpy list
        xywhn = det.boxes.xywhn.cpu().numpy()  # (N,4) normalized
        xyxy  = det.boxes.xyxy.cpu().numpy()   # (N,4) pixels
        names = [det.names[int(cls.item())] for cls in det.boxes.cls.int()]

        for (cx, cy, w, h), (x1, y1, x2, y2), name in zip(xywhn, xyxy, names):
            area_norm = float(w) * float(h)  # Normalized area
            if area_norm >= self.min_area_norm:
                filtered.append({
                    'xyxy': [float(x1), float(y1), float(x2), float(y2)],
                    'name': name
                })
        return filtered

    def track_product(self, detections):
        """
        detections: Ultralytics results list
        return: List of product names confirmed to be added (Product reached 'count == count_threshold')
        """
        add_list = []

        products = self._filter_detection(detections)

        for track in self.track_list:
            track.reset_update_flag()

        for prd in products:
            xyxy = prd['xyxy']
            name = prd['name']
            best_iou = 0
            best_track = None

            # Match with existing track
            for track in self.track_list:
                iou = self._compute_iou(xyxy, track.location)
                if iou>=self.iou_threshold and iou>best_iou:
                    best_iou = iou
                    best_track = track

            # Update existing track
            if best_track:
                best_track.update(xyxy)
                
                if best_track.count == self.count_threshold:
                    add_list.append(name)
                    if DEBUG:
                        print(f'{name} added to cart.')

            # Create new track
            else:
                self.track_list.append(Track(name, xyxy))

        # Remove not updated tracks.
        self.track_list = [t for t in self.track_list if t.update_flag]

        if DEBUG and len(self.track_list) > 0:
            log='tracker list: '
            for track in self.track_list:
                log += f"{track.name}({track.count}), "
            print(log)
        
        return add_list

# Access to results.
# for result in results:
#     xywh = result.boxes.xywh
#     xywhn = result.boxes.xywhn  # normalized
#     xyxy = result.boxes.xyxy
#     xyxyn = result.boxes.xyxyn  # normalized
#     names = [result.names[cls.item()] for cls in result.boxes.cls.int()]
#     confs = result.boxes.conf