import cv, cv2
import sys

import numpy as np

windowName = 'Captured_Image'

img = cv2.imread("images/checkerboard.jpg", cv2.CV_LOAD_IMAGE_COLOR)

cv2.namedWindow(windowName)
cv2.moveWindow(windowName, 0, 0) 
cv2.imshow(windowName, img)

patternsize = (7,7)
(patternfound, corners) = findChessboardCorners(img, patternsize, \
        CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);

if(patternfound)
  cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
    TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

drawChessboardCorners(img, patternsize, Mat(corners), patternfound);

cv2.waitKey(0)
cv2.destroyAllWindows()

