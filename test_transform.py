import cv, cv2
import sys

import numpy as np

srcArr = [-1]*8
point_index = 0

def transformAndDispImage(img, srcArr):
    srcPoints = np.array(srcArr, np.float32).reshape(-1, 2)
    
    mid_x = 300
    inc_x = 30
    mid_y = 400
    inc_y = 30
    dstPoints = np.array([mid_x - inc_x, mid_y + inc_y,\
                          mid_x + inc_x, mid_y + inc_y,\
                          mid_x + inc_x, mid_y - inc_y,\
                          mid_x - inc_x, mid_y - inc_y], np.float32).reshape(-1, 2)

    homoMat = cv2.findHomography(srcPoints, dstPoints)[0]

    res = cv2.warpPerspective(img, homoMat, (600,600))
    cv2.namedWindow('Warped_Image')
    cv2.imshow('Warped_Image', res)
            

def mouseEvent(event, x, y, flags, param):
    global point_index

    if event == 1 and point_index < 4:
        print point_index, ":", x, y

        srcArr[point_index*2] = x
        srcArr[point_index*2 + 1] = y
        point_index += 1

        if point_index == 4:
            transformAndDispImage(img, srcArr)
        

img = cv2.imread("images/tiles2.jpg", cv2.CV_LOAD_IMAGE_COLOR)

# cv2.setMouseCallback("Captured_Image", mouseEvent, 1);

srcPoints = [271,208,398,208,389,151,279,151]
transformAndDispImage(img, srcPoints);

cv2.namedWindow('Captured_Image')
cv2.imshow('Captured_Image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()

