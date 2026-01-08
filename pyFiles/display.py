import cv2
import numpy as np

def displayImg(img, window_name = "image", dim = 800):

    height, width = img.shape[:2]
    scale_ratio = dim / width
    new_dimns = (dim, int(height * scale_ratio))
    img_resized = cv2.resize(img, new_dimns)
    cv2.imshow(window_name, img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
