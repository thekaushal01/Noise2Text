import cv2
from PIL import Image
import numpy as np


def preprocess_image(image: Image.Image) -> np.ndarray:
    img = np.array(image)

    #convert to grayscale depending on what we got
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    elif img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #scale it up so small text doesn't get missed
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    #invert — easyocr reads white text on dark background better
    _, img = cv2.threshold(img, 140, 255, cv2.THRESH_BINARY_INV)

    #clean up any leftover noise
    img = cv2.medianBlur(img, 3)

    return img

