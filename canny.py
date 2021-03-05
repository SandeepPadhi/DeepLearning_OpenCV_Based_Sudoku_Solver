import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('sudoku.jpeg')
img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(img,100,200)

plt.subplot(121),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges)
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
