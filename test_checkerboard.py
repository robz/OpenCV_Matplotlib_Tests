import cv, cv2
import numpy as np

nBoards = 1
boardWidth = 25
boardHeight = 17
nCorners = boardWidth*boardHeight
boardSize = (boardWidth,boardHeight)
successes = 0


def findCorners(image):
   grayImage = cv2.cvtColor(image,cv.CV_BGR2GRAY)
   
   # find chessboard corners
   (found,corners) = cv2.findChessboardCorners(
      image = grayImage,
      patternSize = boardSize,
      flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE
   )
   
   if not found:	
      print "ummm:",corners.shape	
   
   # refine corner locations
   cv2.cornerSubPix(
      image = grayImage,  
      corners = corners,  
      winSize = (11,11),  
      zeroZone = (-1,-1),
      criteria = (cv.CV_TERMCRIT_EPS|cv.CV_TERMCRIT_ITER,30,0.1)  
   )
   
   return corners    
   

def correctDistortion(image):
   
   size = image.shape[1],image.shape[0]
   
   corners = findCorners(image)

   patternPoints = np.zeros( (np.prod(boardSize), 3), np.float32 )
   patternPoints[:,:2] = np.indices(boardSize).T.reshape(-1, 2)
   
   imagePoints = np.array([corners.reshape(-1, 2)])
   objectPoints = np.array([patternPoints])
   cameraMatrix = np.zeros((3, 3))
   distCoefs = np.zeros(4)
   rc,cameraMatrix,distCoeffs,rvecs,tvecs = cv2.calibrateCamera(
      objectPoints,
      imagePoints,
      boardSize,
      cameraMatrix,
      distCoefs
   )
   
   newCameraMatrix,newExtents = cv2.getOptimalNewCameraMatrix(cameraMatrix,distCoeffs,size,0.0)
   
   mapx, mapy = cv2.initUndistortRectifyMap(
      cameraMatrix,
      distCoeffs,
      None,
      cameraMatrix,
      size,
      cv2.CV_32FC1
   )
   newImage = cv2.remap( image, mapx, mapy, cv2.INTER_LANCZOS4 )
   return newImage


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


def correctPerspective(image):
   
   size = image.shape[1],image.shape[0]
   
   corners = findCorners(image)

   # visualise the chessboard corners onto the image
   annotatedImage = image.copy()
   cv2.drawChessboardCorners(annotatedImage,boardSize,corners,1) 
   cv2.imwrite('annotated.jpg',annotatedImage)
   
   rcCorners = corners.reshape(boardHeight,boardWidth,2)
   # now a 3-d array -- a row is new[x] and a column is new [:,x]
   
   # find top left corner point and bottom right corner point (NOT the same as min/max x and y):
   outerPoints = getOuterPoints(rcCorners)
   tl,tr,bl,br = outerPoints
   
   patternSize = np.array([
      np.sqrt(((tr - tl)**2).sum(0)),
      np.sqrt(((bl - tl)**2).sum(0)),
   ])
   
   inQuad = np.array(outerPoints,np.float32)
   
   outQuad = np.array([
      tl,
      tl + np.array([patternSize[0],0.0]),
      tl + np.array([0.0,patternSize[1]]),
      tl + patternSize,
   ],np.float32)
   
   transform = cv2.getPerspectiveTransform(inQuad,outQuad)
   transformed = cv2.warpPerspective(image,transform,size)

   # calculate DPI for the transformed image
   transformedCorners = cv2.perspectiveTransform(corners,transform)
   rcTransformedCorners = transformedCorners.reshape(boardHeight,boardWidth,2)
   outerPoints = getOuterPoints(rcTransformedCorners)
   tl,tr,bl,br = outerPoints
   transformedPatternSize = np.array([
      np.sqrt(((tr - tl)**2).sum(0)),
      np.sqrt(((bl - tl)**2).sum(0)),
   ])
   dpi = (transformedPatternSize / realSizeMM) * 25.4
   print 'dpi before aspect ratio correction',dpi
   
   # correct aspect ratio (stretch the dimension with the lowest resolution so we don't loose data needlessly)
   if dpi[1] > dpi[0]:
      fx = dpi[1]/dpi[0]
      fy = 1.0
      print 'final dots per inch',dpi[1]
   else:
      fx = 1.0
      fy = dpi[0]/dpi[1]
      print 'final dots per inch',dpi[0]
      
   final = cv2.resize(transformed,None,fx=fx,fy=fy)
   
   return final

   
if __name__ == '__main__':
   realSizeMM = np.array([254.0,170.0])
   original = cv2.imread(r'images/distortion.jpg')

   i = correctPerspective(original.copy())
   cv2.imwrite('corrected-perspective-only.jpg',i)
   
   i = correctDistortion(original.copy())
   cv2.imwrite('corrected-distortion-only.jpg',i)
   i = correctPerspective(i)
   cv2.imwrite('corrected-both.jpg',i)
   print 'done'
