import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image


class YoloAksaraLontara:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found: {model_path}")

        self.model = YOLO(model_path)

    # ------------------------------------------------------------
    # Helper: merge close boxes (same as your function)
    # ------------------------------------------------------------
    def merge_boxes(self, boxes, max_gap=25):
        if len(boxes) == 0:
            return []

        boxes = sorted(boxes, key=lambda b: b[0])
        merged = []

        current = boxes[0]
        for b in boxes[1:]:
            if b[0] - current[2] < max_gap:
                current = [
                    min(current[0], b[0]),
                    min(current[1], b[1]),
                    max(current[2], b[2]),
                    max(current[3], b[3]),
                ]
            else:
                merged.append(current)
                current = b

        merged.append(current)
        return merged

    # ------------------------------------------------------------
    # YOLO Detection Only
    # ------------------------------------------------------------
    def det_aksara_from_image(self, image_input, conf=0.75, merge_gap=25):
        """
        Accepts:
        - file path (str)
        - PIL Image
        - numpy ndarray

        Returns:
        - list of boxes in xyxy format: [x1, y1, x2, y2]
        """

        # ----------------------------------------
        # Convert input to numpy RGB
        # ----------------------------------------
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found: {image_input}")
            img = cv2.imread(image_input)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        elif isinstance(image_input, Image.Image):
            img_rgb = np.array(image_input.convert("RGB"))

        elif isinstance(image_input, np.ndarray):
            img_rgb = image_input

        else:
            raise TypeError("Unsupported image type for YOLO detection.")

        # ----------------------------------------
        # Run YOLO detection
        # ----------------------------------------
        results = self.model(img_rgb, conf=conf)

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int).tolist()

        # ----------------------------------------
        # Optional merging
        # ----------------------------------------
        if merge_gap > 0:
            boxes = self.merge_boxes(boxes, max_gap=merge_gap)

        return boxes

# Usage Example
# detector = YoloAksaraLontara(
#     model_path="/content/runs/detect/Buginese_Script_Detection/weights/best.pt"
# )

# boxes = detector.det_aksara_from_image("/content/test-inference-1.jpg")

# print("Detected boxes:", boxes)
