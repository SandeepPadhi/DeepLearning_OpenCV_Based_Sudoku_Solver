import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2 
import numpy as np
from matplotlib import pyplot as plt

sudoku=cv2.imread('sudoku.jpg')
#sudoku = cv2.GaussianBlur(sudoku.copy(), (9, 9), 0)
sudoku=cv2.resize(sudoku,(700,700))
sudoku_gray = cv2.cvtColor(sudoku, cv2.COLOR_BGR2GRAY) 
sudoku_gray = cv2.GaussianBlur(sudoku_gray.copy(), (9, 9), 0)

#ret, sudoku_thresh = cv2.threshold(sudoku_gray, 160, 255, cv2.THRESH_BINARY)
thresh2 = cv2.adaptiveThreshold(sudoku_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)  
#kernel = np.ones((3,3), np.uint8) 
#sudoku_dilation = cv2.dilate(sudoku_thresh, kernel, iterations=2) 


#kernel = np.ones((2,2), np.uint8) 

#sudoku_dilation = cv2.dilate(thresh2, kernel, iterations=1)
#kernel = np.ones((3,3), np.uint8) 
 
#sudoku_erosion = cv2.erode(thresh2, kernel, iterations=1)
#sudoku_erosion=~sudoku_erosion
sudoku_erosion=~thresh2
kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
sudoku_dilate = cv2.dilate(sudoku_erosion, kernel)

contours, hierarchy = cv2.findContours(sudoku_dilate.copy() ,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key=cv2.contourArea, reverse=True)
#polygon = contours[0]
print("contour[0]:{}".format(len(contours[0])))


cv2.drawContours(sudoku, contours, 0, (0, 255, 0), 3) 

cv2.imshow('sudoku',sudoku)
cv2.waitKey(0)
cv2.destroyAllWindows()


"""



import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import numpy as np
import cv2
import math
import time


im = cv2.imread('sudoku.jpg')
im=cv2.resize(im,(700,700))

print("shape:{}".format(len(im)))
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
thresh=~thresh

contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
#polygon = contours[0]
print("contour[0]:{}".format(len(contours[0])))
#cv2.drawContours(im, contours, 0, (0, 255, 0), 3)

def distance(p1,p2):
    return math.sqrt(math.pow(p1[0]-p2[0],2)+math.pow(p1[1]-p2[1],2))


def finddot(p1,p2,p3):
        #print("p1:{} ,p2:{} ,p3:{}".format(p1,p2,p3))
        import math
        l1=(p1[0]-p2[0],p1[1]-p2[1])
        d1=math.sqrt(math.pow(l1[0],2)+math.pow(l1[1],2))
        #print("d1:{}".format(d1))
        l2=(p2[0]-p3[0],p2[1]-p3[1])

        d2=math.sqrt(math.pow(l2[0],2)+math.pow(l2[1],2))
        #print("d2:{}".format(d2))
        #print()
        dot=(l1[0]*l2[0]+l1[1]*l2[1])
        dot=dot/(d1*d2)
        return abs(dot)
polygon=[]
for p in contours[0]:
    polygon.append(p[0])

Ans=[]
for i in range(0,len(polygon)):
    #print("i")
    p1,p2,p3=polygon[i-10],polygon[i],polygon[(i+10)%len(polygon)]
    dot=finddot(p1,p2,p3)
    Ans.append((dot,p2))

Ans.sort(key=lambda x:x[0])
polygon=[Ans[i][1] for i in range(min(len(Ans),20))]

Ans=[]
for i in range(len(polygon)):
    p1,p2,p3=polygon[i-1],polygon[i],polygon[(i+1)%len(polygon)]
    dot=finddot(p1,p2,p3)
    Ans.append((dot,p2))

Ans.sort(key=lambda x:x[0])
#Ans=[Ans[i][1] for i in range(len(Ans))]
i=0
while(i<len(Ans)):
    j=0
    while(j<len(Ans)):
        if i==j:
            j+=1
            continue
        if distance(Ans[i][1],Ans[j][1])<30:
            Ans.pop(j)
        else:
            j+=1
    i+=1
        
Final=[]
i=0



while(len(Ans)==4):
    break

print("Ans:{}".format(Ans))
print("Ans[:4] :{}".format(Ans[:4]))

for i in range(len(Ans[:4])):

    center_coordinates=tuple(Ans[i][1])
    
    radius=1
    color = (255, 0, 0)
    thickness=3
    im = cv2.circle(im, center_coordinates, radius, color, thickness)
Ans=[a[1] for a in Ans[:4]]
Ans=sorted(Ans,key=lambda x:(x[0]))
Ans=sorted(Ans[:2],key=lambda x:x[1]) + sorted(Ans[2:],key=lambda x:x[1])

print("Ans final:{}".format(Ans))
pts1 = np.float32([Ans])
pts2 = np.float32([[0,0],[0,300],[300,0],[300,300]])

M = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(im,M,(9*28,9*28))
print("shape of dst:{}".format(np.shape(dst)))
for i in range(9):
    for j in range(9):
        cv2.imshow('i',dst[i*28:i*28+28,j*28:j*28+28])
        break



#Ans=Ans[:4]
#for i in range(4):
    
cv2.imshow('sudoku',im)
cv2.imshow('per',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()


"""