from ocr.OcrAksaraLontara import OcrAksaraLontara

ocr = OcrAksaraLontara()

ocr_output = ocr.ocr_aksara_from_image("dir_pdf\sample_lontara_1.pdf")

print("ocr output:", ocr_output)