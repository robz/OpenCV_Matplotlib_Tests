import cv, cv2
import sys
import numpy as np

img = cv2.imread("TESTIMAGE.jpg", cv2.CV_LOAD_IMAGE_GRAYSCALE)
output = cv2.threshold(img, 128, 255, cv2.THRESH_TOZERO)[1]

cv2.namedWindow('Display Window')
cv2.imshow('Display Window', output)

cv2.waitKey(0)
cv2.destroyAllWindows()
