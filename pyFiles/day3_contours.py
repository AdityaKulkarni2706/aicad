import display
import numpy as np
import cv2

img = cv2.imread("../images/sample_2.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 115, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print(f"Number of shapes found: {len(contours)}")

# Draw Contours
# -1 means "draw all contours". (0, 255, 0) is Green. 3 is thickness.
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
cv2.imshow("Contours", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

