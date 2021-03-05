import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import numpy as np
import math
import time
#from opencv_try import *
import cv2




def get_Image_data():
    #sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    #import cv2

    im = cv2.imread('sudoku.jpg')
    print("size of im:{}".format(np.shape(im)))
    #im=cv2.resize(im,(400,400))

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
    side=9*28
    pts2 = np.float32([[0,0],[0,side],[side,0],[side,side]])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    dst = cv2.warpPerspective(thresh,M,(side,side))
    dside=28
    #dst = cv2.warpPerspective(im,M,(300,300))
    print("shape of dst:{}".format(np.shape(dst)))
    Image_Data=[]

    for i in range(9):
        for j in range(9):
            pred_img=dst[i*dside:i*dside+dside,j*dside:j*dside+dside]
            #cv2.imshow('i',pred_img)
            #cv2.waitKey(300) #change to your own waiting time 1000 = 1 second 
            Image_Data.append((i,j,pred_img))    



    #Ans=Ans[:4]
    #for i in range(4):
        
    #cv2.imshow('sudoku',im)
    #cv2.imshow('per',dst)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return Image_Data
#get_Image_data()
#print(get_Image_data())