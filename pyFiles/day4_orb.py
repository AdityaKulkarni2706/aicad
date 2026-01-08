import cv2
import numpy as np
import matplotlib.pyplot as plt
import display

img = cv2.imread("../images/sample_2.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def showFeatureDescript(grayImg):
    orb = cv2.ORB_create()
    keypoints = orb.detect(grayImg, None)
    keypoints, descriptors = orb.compute(gray, keypoints)
    img = cv2.drawKeypoints(gray, keypoints, None, color=(0,255,0), flags=0)
    display.displayImg(img, "ORB Keypoints")

showFeatureDescript(gray)
