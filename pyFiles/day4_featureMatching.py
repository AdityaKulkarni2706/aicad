import cv2
import numpy as np
import matplotlib.pyplot as plt
import display

img1 = cv2.imread("../images/cup_1.jpeg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("../images/cup_2.jpeg", cv2.IMREAD_GRAYSCALE)

def matchFeatures(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key = lambda x: x.distance)
    result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(15,10))
    plt.imshow(result)

    plt.title("ORB Feature Matching")
    plt.show()


matchFeatures(img1, img2)
