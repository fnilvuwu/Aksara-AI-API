import os
import numpy as np
from paddleocr import TextRecognition
from PIL import Image


class OcrAksaraLontara:
    def __init__(self, model_dir: str = "dir_ocr_models\\PP-OCRv5_server_rec_infer"):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        self.model = TextRecognition(model_dir=model_dir)

    # ------------------------------------------------------------
    # OCR Aksara Lontara
    # ------------------------------------------------------------
    def ocr_aksara_from_image(self, image_input):
        """
        Accepts:
        - file path (str)
        - PIL Image
        - numpy ndarray
        Returns ONLY rec_text (str)
        """

        # Case 1: file path
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found: {image_input}")
            input_data = image_input

        # Case 2: PIL Image → convert to ndarray
        elif isinstance(image_input, Image.Image):
            input_data = np.array(image_input.convert("RGB"))

        # Case 3: already ndarray
        elif isinstance(image_input, np.ndarray):
            input_data = image_input

        else:
            raise TypeError("Unsupported image type for OCR.")

        # Run OCR
        output = self.model.predict(input=input_data, batch_size=1)

        if not output or not isinstance(output, list):
            return ""

        return output[0].get("rec_text", "")