import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import numpy as np
import math
import time
import matplotlib.pyplot as plt
#from opencv_try import *
import cv2
img=cv2.imread('./lena.jpeg')
img=cv2.resize(img,(320,320))
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
org=img.copy()
vl=145
vh=190
con=8

def findcomponents(img):
    z=np.zeros((16,16))
    for i in range(1,len(img)-1):
        for j in range(1,len(img)-1):
            if con==4 and all(vl<=img[i+di][j+dj]<=vh for di,dj in ((1,0),(-1,0),(0,1),(0,-1))):
                z[i][j]=img[i][j]
            if con==8 and all(vl<=img[i+di][j+dj]<=vh for di,dj in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(-1,1),(1,-1))):
                z[i][j]=img[i][j]
    return z



for i in range(20):
    for j in range(20):
        subimg=img[i*16:i*16+16,j*16:j*16+16]
        z=findcomponents(subimg)
        img[i*16:i*16+16,j*16:j*16+16]=z


plt.subplot(1,2,1)
plt.imshow(org)
plt.subplot(1,2,2)
plt.imshow(img)
plt.show()

