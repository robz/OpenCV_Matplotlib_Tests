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
        bounds = ColorBounds(Cb_min=self.x0, Cr_min=self.y0, Cb_max=self.x1, Cr_max=self.y1)

class ColorBounds(object):
    def __init__(self, Cb_min=0, Cb_max=255, Cr_min=0, Cr_max=255, Y_min=0, Y_max=255):
        self.Pb_min = Cb_min/255.0
        self.Pb_max = Cb_max/255.0
        self.Pr_min = Cr_min/255.0
        self.Pr_max = Cr_max/255.0
        self.Yp_min = Y_min/255.0
        self.Yp_max = Y_max/255.0
        print(self.rgbBound(self.Yp_min, self.Pb_min, self.Pr_min, Kb, Kr))
        print(self.rgbBound(self.Yp_max, self.Pb_max, self.Pr_max, Kb, Kr))

    def rgbBound(self, y, p, q, K, J):
        b = y - 2*(-1 + J)*p 
        r = y - 2*(-1 + K)*q  
        g = (b*J + K*r - y)/(-1 + J + K)
        return (r, g, b)
    

def plotIn2d(xs, ys, xaxis, yaxis, colors):
    ax = plt.gca()

    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)

    ax.set_xlabel(xaxis, fontsize=20)
    ax.set_ylabel(yaxis, fontsize=20)
    ax.grid(True)

    ax.scatter(xs, ys, c=colors, s=20, alpha=1)
    a = Annotate(ax)

    plt.show()

    return a

def convertToChannelsAndPlot(img):
    arr2d = np.asarray(img)
    height = len(arr2d)
    width = len(arr2d[0])

    print(width*height)

    ys = [None]*NUM_PIXELS
    cbs = [None]*NUM_PIXELS
    crs = [None]*NUM_PIXELS
    colors = [[None]*3]*NUM_PIXELS  

    for index in range(NUM_PIXELS):
        cell = random.choice(random.choice(arr2d))

        blue = cell[0]/256.0
        green = cell[1]/256.0
        red = cell[2]/256.0
        colors[index] = [red, green, blue]
        
        y = Kr*red + (1 - Kr - Kb)*green + Kb*blue
        pb = .5*(blue - y)/(1 - Kb)
        pr = .5*(red - y)/(1 - Kr)
        
        ys[index] = 16 + 219*y
        cbs[index] = 128 + 224*pb
        crs[index] = 128 + 224*pr


    print("done processing!")

    # plotIn3d(cbs, crs, ys, colors)
    a = plotIn2d(cbs, crs, "Cb", "Cr", colors)

    return a

img = cv2.imread("becon_plain.png", cv2.CV_LOAD_IMAGE_COLOR)

print(img.shape)

cv2.namedWindow('Display Window')
cv2.imshow('Display Window', img)

a = convertToChannelsAndPlot(img)
bounds = ColorBounds(Cb_min=a.x0, Cr_min=a.y0, Cb_max=a.x1, Cr_max=a.y1)

cv2.waitKey(0)
cv2.destroyAllWindows()







