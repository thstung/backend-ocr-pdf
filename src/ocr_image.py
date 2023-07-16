from PIL import Image
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import time
from src.init_models import viet_ocr, paddle_ocr

# def sort_lines(list_line):

#     if len(list_line) == 0:
#         return list_line
#     elif len(list_line[0]) == 0:
#         return list_line
#     # Sắp xếp mỗi bbox trong mỗi line theo y0, x0
#     for line in list_line:
#         line.sort(key=lambda bbox: (bbox[0][1], bbox[0][0]), reverse=True)

#     # Sắp xếp các line trong s theo y0 min trong line đó
#     list_line.sort(key=lambda line: line[0][0][1], reverse=False)

#     return list_line

def ocr_image(image):
    image_array = np.array(image)
    result = paddle_ocr.ocr(image_array, rec=False)
    # result = sort_lines(list_lines)
    list_text = []
    for line in result:
        for bbox in line:
            left = bbox[0][0]
            top = bbox[0][1]
            right = bbox[2][0]
            bot = bbox[2][1]
            im_crop = image.crop((left, top, right, bot))
            recognized_text = viet_ocr.predict(im_crop)
            list_text.insert(0, recognized_text)
    text = ' '.join(list_text)
    return text

if __name__ == "__main__":
    image_path = "../fpt/detected_pages/fpt_page_0_text.jpg"
    image_pil = Image.open(image_path)
    detection = ocr_image(image_pil)
    print(detection)