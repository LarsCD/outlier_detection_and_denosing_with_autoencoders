import cv2
import numpy as np

def anonymize_center(image, factor=3.0, region_ratio=0.5):
    # image-afmetingen
    (h, w) = image.shape[:2]

    # grootte van het middengebied (bijv. 40% van hoogte/breedte)
    rw = int(w * region_ratio)
    rh = int(h * region_ratio)

    # coördinaten van het midden
    x1 = (w - rw) // 2
    y1 = (h - rh) // 2
    x2 = x1 + rw
    y2 = y1 + rh

    # uitsnijden van middengebied
    center_region = image[y1:y2, x1:x2]

    # blur instellen zoals jouw functie dat deed
    kW = int(rw / factor)
    kH = int(rh / factor)
    if kW % 2 == 0: kW -= 1
    if kH % 2 == 0: kH -= 1

    blurred = cv2.GaussianBlur(center_region, (kW, kH), 0)

    # geblurd deel terugplaatsen
    output = image.copy()
    output[y1:y2, x1:x2] = blurred

    return output
