import cv2
import numpy as np
import display

img = cv2.imread("../images/square.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def showHarrisCorners(grayImg):
    grayImg = np.float32(grayImg)
    dst = cv2.cornerHarris(grayImg, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)
    img[dst > (0.01 * dst.max())] = [0, 0, 255] #format is BGR
    display.displayImg(img, "Harris Corners (Red)")


showHarrisCorners(gray)
