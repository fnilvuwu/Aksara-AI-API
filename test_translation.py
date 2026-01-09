import os
from processor.ProcessorAksaraLontara import ProcessorAksaraLontara

# Initialize processor
processor = ProcessorAksaraLontara()

# ------------------------------------------------------------
# TEST 1: Translate pure text
# ------------------------------------------------------------
print("\n=== TEST 1: TEXT TRANSLATION ===")

sample_text = "ᨕᨁ ᨀᨑᨙᨅᨑ"
try:
    result_text = processor.generate_translation_from_text(sample_text)
    print(result_text)
    print(type(result_text))
except Exception as e:
    print("Error:", e)


# ------------------------------------------------------------
# TEST 2: Translate from an image
# ------------------------------------------------------------
print("\n=== TEST 2: IMAGE OCR TRANSLATION ===")

image_path = "dir_images/test1.png"

try:
    result_image = processor.generate_translation_from_image(image_path)
    print(result_image)
    print(type(result_image))
except Exception as e:
    print("Error:", e)


# ------------------------------------------------------------
# TEST 3: Translate from a PDF
# ------------------------------------------------------------
print("\n=== TEST 3: PDF OCR TRANSLATION ===")

pdf_path = "dir_pdf/test1.pdf"

try:
    result_pdf = processor.generate_translation_from_pdf(pdf_path)
    print(result_pdf)
    print(type(result_pdf))
except Exception as e:
    print("Error:", e)

print("\n=== TEST FINISHED ===")
