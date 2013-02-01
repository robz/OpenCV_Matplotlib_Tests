import cv,cv2
import sys
import numpy as np

# (cols, rows) of interior corners (so an 8x8 grid would have (7, 7))
boardSize = (7, 7)
imageSrcName = 'images/phone_board2.jpg' 

def getOuterPoints(rcCorners):
   tl = rcCorners[0,0]
   tr = rcCorners[0,-1]
   bl = rcCorners[-1,0]
   br = rcCorners[-1,-1]
   if tl[0] > tr[0]:
      tr,tl = tl,tr
      br,bl = bl,br
   if tl[1] > bl[1]:
      bl,tl=tl,bl
      br,tr=tr,br
   return (tl,tr,bl,br)

def transformImage(img, corners, boardSize):
    size = img.shape[1],img.shape[0]

    rcCorners = corners.reshape(boardSize[0], boardSize[1], 2)

    outerPoints = getOuterPoints(rcCorners)
    tl,tr,bl,br = outerPoints

    patternSize = np.array([
        np.sqrt(((tr - tl)**2).sum(0)),
        np.sqrt(((bl - tl)**2).sum(0)),
    ])
   
    inQuad = np.array(outerPoints, np.float32)

    outQuad = np.array([
        tl,
        tl + np.array([patternSize[0],0.0]),
        tl + np.array([0.0,patternSize[1]]),
        tl + patternSize,
    ],np.float32)

    transform = cv2.getPerspectiveTransform(inQuad, outQuad)
    transformed = cv2.warpPerspective(img, transform, size)

    return transformed

def dispImages(img):
    (found, corners) = cv2.findChessboardCorners(
        img,
        boardSize,	
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if not found:
        print "SHIT IS FUCKED UP"
        return;
        
    cv2.drawChessboardCorners(img, boardSize, corners, found)
    transformed = transformImage(img, corners, boardSize)

    cv2.imshow('Plain', img)
    cv2.imshow('Transformed', transformed)


img = cv2.imread(
    imageSrcName,
    cv2.CV_LOAD_IMAGE_COLOR
)

cv2.namedWindow('Plain')
cv.MoveWindow('Plain', 0, 0)

cv2.namedWindow('Transformed')
cv.MoveWindow('Transformed', img.shape[1], 0)

dispImages(img)

cv2.waitKey(0)                          
cv2.destroyAllWindows() 

































