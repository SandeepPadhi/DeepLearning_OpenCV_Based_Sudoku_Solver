"""
1. Take the Lena image shown below with input in size 64×64 and zoom the  image to 128×128 
 using bilinear interpolation and show both the images in a  single window using subplot in
 Matlab or any other equivalent functions in  python. (Do not use the inbuilt functions to 
 zoom and interpolate the  image. Create your own bilinear interpolation function to interpolate 
 the  image. Can use inbuilt functions to read and show the image) [10] 


"""
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import numpy as np
import math
import time
import matplotlib.pyplot as plt
#from opencv_try import *
import cv2
img=cv2.imread('./lena.jpeg')
img=cv2.resize(img,(64,64))
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#cv2.imshow('sudoku',im)
def zoomcol(img):
    z=np.zeros((64,128))
    for i in range(64):
        for j in range(64):
            z[i][2*j]=img[i][j]

    for i in range(64):
        for j in range(63):
            v=(int(img[i][j])+int(img[i][j+1]))//2
            z[i][2*j+1]=v

    return z.astype('uint8')

def zoomrow(img):
    z=np.zeros((128,128))
    for i in range(64):
        for j in range(128):
            z[2*i][j]=img[i][j]
    for i in range(63):
        for j in range(128):
            v=(int(img[i][j])+int(img[i+1][j]))//2
            z[2*i+1][j]=v
    return z.astype('uint8')


zoom=zoomcol(img.copy())
zoom=zoomrow(zoom.copy())
plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(zoom)
plt.show()

