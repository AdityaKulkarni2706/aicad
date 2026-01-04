import cv2
import numpy as np

image_path = "../images/sample.jpg"
img = cv2.imread(image_path)
print(f"image shape : {img.shape}")

height, width = img.shape[:2]
scale_ratio = 800 / width
new_dimns = (800, int(height * scale_ratio))
img_resized = cv2.resize(img, new_dimns)
print(f"New dimensions : {img_resized.shape}")
