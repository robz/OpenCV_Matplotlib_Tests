import cv, cv2
import sys

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle

import numpy as np
import random

NUM_PIXELS = 10000
Kb = .0722
Kr = .2126

CONE_THREASHOLD_BOUNDS = (0.71484374999999978, 0.99609375000000011, 0.15625, 0.3515625, 0.085937499999999958, 0.29687500000000011)

BARREL_THREASHOLD_BOUNDS = (0.66796875, 0.99609375000000022, 0.10156250000000006, 0.62109375, 0.27343749999999994, 0.71484374999999989)

def plotIn3d(xs, ys, zs, colors):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.scatter(xs, ys, zs, c=colors, alpha=1)

    ax.legend()
    ax.set_xlim3d(0, 255)
    ax.set_ylim3d(0, 255)
    ax.set_zlim3d(0, 255)

    ax.set_xlabel('Cb')
    ax.set_ylabel('Cr')
    ax.set_zlabel('Y')

    plt.show()

class Annotate(object):
    def __init__(self, ax):
        self.ax = ax
        self.rect = Rectangle((0,0), 1, 1, fill=False)
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.ax.add_patch(self.rect)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)

    def on_press(self, event):
        print 'press'
        self.x0 = event.xdata
        self.y0 = event.ydata

    def on_release(self, event):
        print 'release'
        self.x1 = event.xdata
        self.y1 = event.ydata
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy((self.x0, self.y0))
        self.ax.figure.canvas.draw()
        self.get_graph_bound()
        #self.bounds = ColorBounds(Cb_min=self.x0, Cr_min=self.y0, Cb_max=self.x1, Cr_max=self.y1)
        self.bounds = ColorBounds(Cr_min=self.x0, Yp_min=self.y0, Cr_max=self.x1, Yp_max=self.y1)
        
    def get_graph_bound(self):
        print(self.x0, self.y0, self.x1, self.y1)
        # return (x0, y0, x1, y1)


class ColorBounds(object):
    def __init__(self, Cb_min=16, Cb_max=235, Cr_min=16, Cr_max=235, Yp_min=16, Yp_max=235):
        self.Cb_min = self.saturate(Cb_min)
        self.Cb_max = self.saturate(Cb_max)

        self.Cr_min = self.saturate(Cr_min)
        self.Cr_max = self.saturate(Cr_max)

        self.Yp_min = self.saturate(Yp_min)
        self.Yp_max = self.saturate(Yp_max)
        

        self.red_min = 255
        self.blue_min = 255
        self.green_min = 255
        self.red_max = 0
        self.blue_max = 0
        self.green_max = 0

    def saturate(self, val):
        if val < 16:
            return 16
        elif val > 235:
            return 235
        return int(val)

    def isWithin(self, Y, Cb, Cr):
        return self.Yp_min <= Y and Y <= self.Yp_max and \
               self.Cr_min <= Cr and Cr <= self.Cr_max and \
               self.Cb_min <= Cb and Cb <= self.Cb_max
            
    def change(self, Y, Cb, Cr):
        (red, blue, green) = self.toRGB(Y, Cb, Cr)

        if red < self.red_min:
            self.red_min = red
        if red > self.red_max:
            self.red_max = red
        
        if blue < self.blue_min:
            self.blue_min = blue
        if blue > self.blue_max:
            self.blue_max = blue
        
        if green < self.green_min:
            self.green_min = green
        if green > self.green_max:
            self.green_max = green

        """
        if Y < self.Yp_min:
            self.Yp_min = Y
        if Y > self.Yp_max:
            self.Yp_max = Y
        
        if Cb < self.Cb_min:
            self.Cb_min = Cb
        if Cb > self.Cb_max:
            self.Cb_max = Cb
        
        if Cr < self.Cr_min:
            self.Cr_min = Cr
        if Cr > self.Cr_max:
            self.Cr_max = Cr
        """

    def rgbBounds(self):
        return (self.red_min, self.red_max, \
                self.green_min, self.green_max, \
                self.blue_min, self.blue_max)

    def toRGB(self, Yp, Cb, Cr):
        Y = (Yp - 16)/219.0
        Pb = (Cb - 128)/224.0
        Pr = (Cr - 128)/224.0

        blue = Y - 2*Pb*(Kb - 1) 
        red = Y - 2*Pr*(Kr - 1)
        green = (blue*Kb + Kr*red - Y)/(Kb + Kr - 1)

        return (red, green, blue)
    

def plotIn2d(xs, ys, xaxis, yaxis, colors):
    ax = plt.gca()

    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)

    ax.set_xlabel(xaxis, fontsize=20)
    ax.set_ylabel(yaxis, fontsize=20)
    ax.grid(True)

    ax.scatter(xs, ys, c=colors, s=20, alpha=1)
    annotate = Annotate(ax)

    plt.show()

    return annotate.bounds

def convertToChannelsAndPlotAndComputeBounds(img):
    arr2d = np.asarray(img)
    height = len(arr2d)
    width = len(arr2d[0])

    print(width*height)

    ys = [None]*NUM_PIXELS
    cbs = [None]*NUM_PIXELS
    crs = [None]*NUM_PIXELS
    colors = [[None]*3]*NUM_PIXELS  

    print(Kb, Kr)
    for index in range(NUM_PIXELS):
        cell = random.choice(random.choice(arr2d))

        blue = cell[0]/256.0
        green = cell[1]/256.0
        red = cell[2]/256.0

        colors[index] = [red, green, blue]
        
        Y = Kr*red + (1 - Kr - Kb)*green + Kb*blue
        Pb = .5*(blue - Y)/(1 - Kb)
        Pr = .5*(red - Y)/(1 - Kr)
        
        ys[index] = 16 + 219*Y
        cbs[index] = 128 + 224*Pb
        crs[index] = 128 + 224*Pr

    print("done processing!")

    #plotIn3d(cbs, crs, ys, colors)
    colorbounds = plotIn2d(crs, ys, "Cr", "Y", colors)
    

    pixelbounds = ColorBounds()
    for index in range(NUM_PIXELS):
        if colorbounds.isWithin(ys[index], cbs[index], crs[index]):
            pixelbounds.change(ys[index], cbs[index], crs[index])

    return pixelbounds
    
def runBoundedThreshold(img, bounds):
    split_channels = cv2.split(img)
    blue_img = cv2.split(img)[0]
    green_img = cv2.split(img)[1]
    red_img = cv2.split(img)[2]

    flag, redLowerBinImg = cv2.threshold(red_img, bounds[0]*255, 1, cv2.THRESH_BINARY)
    flag, redUpperBinImg = cv2.threshold(red_img, bounds[1]*255, 1, cv2.THRESH_BINARY_INV)
    flag, greenLowerBinImg = cv2.threshold(green_img, bounds[2]*255, 1, cv2.THRESH_BINARY)
    flag, greenUpperBinImg = cv2.threshold(green_img, bounds[3]*255, 1, cv2.THRESH_BINARY_INV)
    flag, blueLowerBinImg = cv2.threshold(blue_img, bounds[4]*255, 1, cv2.THRESH_BINARY)
    flag, blueUpperBinImg = cv2.threshold(blue_img, bounds[5]*255, 1, cv2.THRESH_BINARY_INV)

    mergedImg = cv2.multiply(blueUpperBinImg,\
                 cv2.multiply(blueLowerBinImg,\
                  cv2.multiply(greenUpperBinImg,\
                   cv2.multiply(greenLowerBinImg,\
                    cv2.multiply(redUpperBinImg, redLowerBinImg)))), scale=255)

    return mergedImg


img = cv2.imread("cones_plus_shadowed_barrel.jpg", cv2.CV_LOAD_IMAGE_COLOR)

#bounds = convertToChannelsAndPlotAndComputeBounds(img).rgbBounds()
#print bounds

#cv2.namedWindow('Barrels')
#cv2.imshow('Barrels', runBoundedThreshold(img, bounds))

cv2.namedWindow('Normal Image')
cv2.imshow('Normal Image', img)

cv2.namedWindow('Cones')
cv2.imshow('Cones', runBoundedThreshold(img, CONE_THREASHOLD_BOUNDS))

cv2.namedWindow('Barrels')
cv2.imshow('Barrels', runBoundedThreshold(img, BARREL_THREASHOLD_BOUNDS))

cv2.waitKey(0)
cv2.destroyAllWindows()















