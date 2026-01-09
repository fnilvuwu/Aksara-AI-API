import os
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

class OcrAksaraLontara:
    def __init__(
        self,
        onnx_model_path: str,
        dict_path: str,
        img_height: int = 48,
        max_width: int = 320,
        use_gpu: bool = False,
    ):
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")

        providers = ["CPUExecutionProvider"]
        if use_gpu:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            onnx_model_path,
            providers=providers
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.img_height = img_height
        self.max_width = max_width

        # Load dictionary (must match training)
        with open(dict_path, "r", encoding="utf-8") as f:
            chars = [line.strip() for line in f]

        # CTC blank at index 0
        self.characters = [""] + chars

    # ------------------------------------------------------------
    # Preprocess (PaddleOCR-compatible)
    # ------------------------------------------------------------
    def preprocess(self, image_input):
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found: {image_input}")
            img = Image.open(image_input)

        elif isinstance(image_input, Image.Image):
            img = image_input

        elif isinstance(image_input, np.ndarray):
            img = image_input

        else:
            raise TypeError("Unsupported image type for OCR.")

        if isinstance(img, Image.Image):
            img = img.convert("RGB")
            img = np.array(img)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        h, w, _ = img.shape
        ratio = w / float(h)
        new_w = int(self.img_height * ratio)
        new_w = min(new_w, self.max_width)

        img = cv2.resize(img, (new_w, self.img_height))
        img = img.astype("float32") / 255.0
        img = (img - 0.5) / 0.5
        img = img.transpose(2, 0, 1)  # CHW
        img = np.expand_dims(img, axis=0)

        return img

    # ------------------------------------------------------------
    # CTC Decode
    # ------------------------------------------------------------
    def ctc_decode(self, preds):
        idxs = preds.argmax(axis=2)[0]

        last_idx = -1
        text = []

        for idx in idxs:
            if idx != last_idx and idx != 0:
                if idx < len(self.characters):
                    text.append(self.characters[idx])
            last_idx = idx

        return "".join(text)

    # ------------------------------------------------------------
    # OCR Aksara Lontara (ONNX)
    # ------------------------------------------------------------
    def ocr_aksara_from_image(self, image_input):
        input_tensor = self.preprocess(image_input)

        preds = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )[0]

        return self.ctc_decode(preds)
