import os
import json
from io import BytesIO
from dotenv import load_dotenv

from PIL import Image
import pypdfium2 as pdfium
import google.generativeai as genai

from prompts.PromptAksaraLontara import PromptAksaraLontara
from ocr.OcrAksaraLontara import OcrAksaraLontara

# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Missing GEMINI_API_KEY in environment variables.")

genai.configure(api_key=api_key)


class ProcessorAksaraLontara:

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self.model = genai.GenerativeModel(self.model_name)
        self.ocr = OcrAksaraLontara()   # Initialize once

    # ------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------
    @staticmethod
    def _clean_json_text(text: str) -> str:
        """
        Remove markdown fences and ensure JSON only.
        """
        text = text.strip()

        # Remove markdown fences ```json ... ```
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [line for line in lines if not line.startswith("```")]
            text = "\n".join(lines).strip()

        return text

    @staticmethod
    def _parse_json(text: str) -> dict:
        """
        Convert cleaned JSON string into dict safely.
        """
        try:
            return json.loads(text)
        except Exception:
            # Fallback to empty JSON-safe structure
            return {"aksara": "", "latin": "", "indonesia": ""}

    @staticmethod
    def _pdf_to_images(pdf_path: str):
        """Yield PIL images for each PDF page."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pdf = pdfium.PdfDocument(pdf_path)
        for i in range(len(pdf)):
            page = pdf.get_page(i)
            yield page.render(scale=2).to_pil()

    def _call_model_json(self, content):
        """Call Gemini + return parsed JSON as dict."""
        response = self.model.generate_content(content)
        cleaned = self._clean_json_text(response.text)
        return self._parse_json(cleaned)

    def _translate_text(self, aksara_text: str, model: str = None) -> dict:
        """Central translation logic for text."""
        if model:
            self.model = genai.GenerativeModel(model)

        prompt = PromptAksaraLontara.prompt_translate_text(aksara_text)
        return self._call_model_json(prompt)

    # ------------------------------------------------------------
    # TEXT
    # ------------------------------------------------------------
    def generate_translation_from_text(self, text: str, model: str = None) -> dict:
        return self._translate_text(text, model)

    # ------------------------------------------------------------
    # IMAGE (OCR -> Text -> Gemini)
    # ------------------------------------------------------------
    def generate_translation_from_image(self, image_path: str, model: str = None) -> dict:
        aksara_text = self.ocr.ocr_aksara_from_image(image_path)
        return self._translate_text(aksara_text, model)

    # ------------------------------------------------------------
    # PDF (OCR per page -> merge text -> Gemini)
    # ------------------------------------------------------------
    def generate_translation_from_pdf(self, pdf_path: str, model: str = None) -> dict:
        pages = list(self._pdf_to_images(pdf_path))
        if not pages:
            return {"aksara": "", "latin": "", "indonesia": ""}

        ocr_results = []

        for img in pages:
            # Convert PIL image → bytes (PNG) for OCR if needed
            buf = BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)

            # OCR expects file path OR ndarray — we use ndarray via numpy array
            np_img = Image.open(buf)
            text = self.ocr.ocr_aksara_from_image(np_img)
            ocr_results.append(text)

        combined_text = "\n".join(ocr_results).strip()
        return self._translate_text(combined_text, model)