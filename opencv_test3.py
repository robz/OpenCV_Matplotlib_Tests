import cv, cv2



img = cv.imread("cones_plus_shadowed_barrel.jpg", cv2.CV_LOAD_IMAGE_COLOR)
cv.namedWindow('Normal Image')
cv.imshow('Normal Image', img)
hsv_img = img
cv.CvtColor(img,hsv_img,cv.CV_RGB2HSV)
cv.namedWindow('HSV Image')
cv.imshow('HSV Image', hsv_img)


cv.waitKey(0)
cv.destroyAllWindows()



