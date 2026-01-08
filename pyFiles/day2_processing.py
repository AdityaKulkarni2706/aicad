import cv2
import numpy as np

def displayImg(img, window_name = "image"):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_path = "../images/sample.jpg"
img = cv2.imread(image_path)
print(f"image shape : {img.shape}")


#Resizing

height, width = img.shape[:2]
scale_ratio = 800 / width
new_dimns = (800, int(height * scale_ratio))
img_resized = cv2.resize(img, new_dimns)
print(f"New dimensions : {img_resized.shape}")


#Cvt to grayscale
gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

#Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5,5), 0)
_, thresholded_img = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

#Canny edge detection
edges = cv2.Canny(blurred, 100, 150)


displayImg(edges)





