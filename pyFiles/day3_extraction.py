import cv2
import numpy as np

img_path = "../images/sample_2.png"

def displayImg(img, window_name = "image", dim = 800):

    height, width = img.shape[:2]
    scale_ratio = dim / width
    new_dimns = (dim, int(height * scale_ratio))
    img_resized = cv2.resize(img, new_dimns)
    cv2.imshow(window_name, img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def photo_to_sketch(img_path, output_path="", k = 23):
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv_img = 255 - gray_img
    blurred_img = cv2.GaussianBlur(inv_img, (k,k), 0)
    inv_blurred_img = 255 - blurred_img
    sketch_img = cv2.divide(gray_img, inv_blurred_img, scale = 256.0)

    displayImg(sketch_img)

photo_to_sketch(img_path, 101)



