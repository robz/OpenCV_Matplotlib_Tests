import cv, cv2
import sys

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np
import random

NUM_PIXELS = 2000
Kb = .0722
Kr = .2126

def readImage():
    img = cv2.imread("becon_plain.png", cv2.CV_LOAD_IMAGE_COLOR)

    print(img.shape)

    cv2.namedWindow('Display Window')
    cv2.imshow('Display Window', img)

    return img

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

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.scatter(cbs, crs, ys, c=colors)

    ax.legend()
    ax.set_xlim3d(0, 255)
    ax.set_ylim3d(0, 255)
    ax.set_zlim3d(0, 255)

    ax.set_xlabel('Cb')
    ax.set_ylabel('Cr')
    ax.set_zlabel('Y')

    plt.show()


convertToChannelsAndPlot(readImage())

cv2.waitKey(0)
cv2.destroyAllWindows()
