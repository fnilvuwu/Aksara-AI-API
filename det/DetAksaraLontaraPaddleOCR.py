import os
import numpy as np
from paddleocr import TextDetection
from PIL import Image


class DetAksaraLontara:
    def __init__(self, model_dir: str = "dir_ocr_models\\PP-OCRv5_server_det_infer"):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        # Load detection model ONLY from model_dir
        self.model = TextDetection(model_dir=model_dir)

    # ------------------------------------------------------------
    # Detection Aksara Lontara
    # ------------------------------------------------------------
    def det_aksara_from_image(self, image_input):
        """
        Accepts:
        - file path (str)
        - PIL Image
        - numpy ndarray

        Returns:
        - detection result objects (list of OCRResult)
        """

        # Case 1: file path
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found: {image_input}")
            input_data = image_input

        # Case 2: PIL Image → convert to ndarray
        elif isinstance(image_input, Image.Image):
            input_data = np.array(image_input.convert("RGB"))

        # Case 3: numpy array
        elif isinstance(image_input, np.ndarray):
            input_data = image_input

        else:
            raise TypeError("Unsupported image type for detection.")

        # Run detection
        output = self.model.predict(input=input_data, batch_size=1)

        return output   # return list of OCRResult objects

# Usage Example
# detector = DetAksaraLontara()

# output = detector.det_aksara_from_image("general_ocr_001.png")

# for res in output:
#     res.print()
#     res.save_to_img(save_path="./output/")
#     res.save_to_json(save_path="./output/res.json")
