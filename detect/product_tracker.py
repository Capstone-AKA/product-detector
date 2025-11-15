import numpy as np
from config_env import DEBUG
from scipy.optimize import linear_sum_assignment  # Hungarian algorithm

class Track:
    __slots__ = ("name", "location", "count", "missed")
    
    def __init__(self, name, location):
        # Initialize a new tracking entry
        self.name = name
        self.location = location  # bbox: [x1,y1,x2,y2]
        self.count = 1            # consecutive matched frames
        self.missed = 0           # missed counts in consecutive frames

    def update(self, location):
        # Update track with a new detection
        self.location = location
        self.count += 1
        self.missed = 0

    def miss(self):
        # Count missed
        self.missed+=1


class ProductTracker:
    def __init__(self, iou_threshold=0.6, count_threshold=4, miss_threshold=2, min_area_norm=0.4):
        # Threshold settings
        self.iou_threshold = float(iou_threshold)
        self.count_threshold = int(count_threshold)
        self.miss_threshold = int(miss_threshold)
        self.min_area_norm = float(min_area_norm)

        # List of Track objects
        self.track_list = []

    def _filter_detection(self, detections):
        """
        Filter detections by normalized area
        Returns a list of dicts: [{ 'xyxy': [...], 'name': str }, ...]
        """

        filtered = []
        if not detections:
            return filtered

        det = detections[0]
        if det.boxes is None or len(det.boxes) == 0:
            return filtered

        # Convert model tensors to numpy arrays
        xyxy  = det.boxes.xyxy.cpu().numpy()   # pixel coordinates
        names = [det.names[int(cls.item())] for cls in det.boxes.cls.int()]

        H, W = det.orig_shape
        x1 = xyxy[:, 0]
        y1 = xyxy[:, 1]
        x2 = xyxy[:, 2]
        y2 = xyxy[:, 3]

        wn = (x2 - x1) / W
        hn = (y2 - y1) / H
        area_norm = wn * hn

        mask = area_norm >= self.min_area_norm

        # Filter by area
        for (bx, name, ok) in zip(xyxy, names, mask):
            if not ok:
                continue

            filtered.append({
                'xyxy': bx[:4],
                'name': name
            })
        return filtered

    def _compute_iou_matrix(self, track_boxes, prod_boxes):
        """
        Compute IoU between all tracks and all detections.
        track_boxes: (T,4)
        prod_boxes:  (P,4)
        Returns IoU matrix of shape (T, P)
        """

        tb = track_boxes[:, None, :]   # (T,1,4)
        pb = prod_boxes[None, :, :]    # (1,P,4)

        xA = np.maximum(tb[..., 0], pb[..., 0])
        yA = np.maximum(tb[..., 1], pb[..., 1])
        xB = np.minimum(tb[..., 2], pb[..., 2])
        yB = np.minimum(tb[..., 3], pb[..., 3])

        interW = np.maximum(0.0, xB - xA)
        interH = np.maximum(0.0, yB - yA)
        interArea = interW * interH

        areaT = np.maximum(0.0, (tb[..., 2] - tb[..., 0])) * np.maximum(0.0, (tb[..., 3] - tb[..., 1]))
        areaP = np.maximum(0.0, (pb[..., 2] - pb[..., 0])) * np.maximum(0.0, (pb[..., 3] - pb[..., 1]))

        denom = areaT + areaP - interArea + 1e-6
        return interArea / denom  # IoU matrix (T,P)
    
    def _log(self, add_list):
        if not DEBUG:
            return

        if self.track_list:
            log_tl = "Track["
            for track in self.track_list:
                log_tl += f"{track.name}({track.count},{track.missed}), "
            print(log_tl+"]")
        else:
            print("no detection, clear track list")

        log_al = "Added["
        if add_list:
            for add in add_list:
                log_al += f"{add}, "
            print(log_al+"]")

    def track_product(self, detections):
        """
        Perform global track-detection assignment using Hungarian algorithm.
        Returns: list of product names that reached count_threshold.
        """

        add_list = []

        # 1. Filter detections
        products = self._filter_detection(detections)

        T = len(self.track_list)
        P = len(products)
        
        # Case: no detections
        if P == 0:
            for track in self.track_list:
                track.miss()
            self.track_list = [t for t in self.track_list if t.missed <= self.miss_threshold]
            self._log(add_list)
            return add_list

        # Case: no existing tracks -> create new ones for all detections
        if T == 0:
            for prd in products:
                xyxy = np.array(prd['xyxy'], dtype=np.float32)
                self.track_list.append(Track(prd['name'], xyxy))
            self._log(add_list)
            return add_list

        # 3. Build IoU matrix
        track_boxes = np.array([t.location for t in self.track_list], dtype=np.float32)
        prod_boxes  = np.array([p['xyxy'] for p in products], dtype=np.float32)

        iou_matrix = self._compute_iou_matrix(track_boxes, prod_boxes)

        # 4. Hungarian assignment (minimizes cost)
        cost_matrix = 1.0 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_tracks = set()
        matched_products = set()

        # 5. Apply assignments with IoU threshold
        for t_idx, p_idx in zip(row_ind, col_ind):
            iou = iou_matrix[t_idx, p_idx]
            if iou < self.iou_threshold:
                continue

            track = self.track_list[t_idx]
            xyxy = prod_boxes[p_idx]

            track.update(xyxy)
            matched_tracks.add(t_idx)
            matched_products.add(p_idx)

            if track.count == self.count_threshold:
                add_list.append(track.name)

        # 6. Unmatched tracks -> missed track
        for t_idx, track in enumerate(self.track_list):
            if t_idx in matched_tracks:
                continue
            track.miss()

        # 7. Unmatched detections -> new tracks
        for p_idx, prd in enumerate(products):
            if p_idx in matched_products:
                continue
            xyxy = prod_boxes[p_idx]
            self.track_list.append(Track(prd['name'], xyxy))

        # 8. Remove tracks that were missed for a while
        self.track_list = [t for t in self.track_list if t.missed <= self.miss_threshold]

        self._log(add_list)
        
        return add_list
