import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import numpy as np
import cv2
import math
import time
from opencv_try import get_Image_data





class CNN(nn.Module):
    def __init__(self, lr, epochs, batch_size, num_classes=10):
        super(CNN, self).__init__()
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.loss_history = []
        self.acc_history = []
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.bn3 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, 3)
        self.bn6 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(2)

        input_dims = self.calc_input_dims()

        self.fc1 = nn.Linear(input_dims, self.num_classes)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        self.loss = nn.CrossEntropyLoss()
        self.to(self.device)
        #self.get_data()

    def calc_input_dims(self):
        batch_data = T.zeros((1, 1, 28, 28))
        batch_data = self.conv1(batch_data)
        #batch_data = self.bn1(batch_data)
        batch_data = self.conv2(batch_data)
        #batch_data = self.bn2(batch_data)
        batch_data = self.conv3(batch_data)

        batch_data = self.maxpool1(batch_data)
        batch_data = self.conv4(batch_data)
        batch_data = self.conv5(batch_data)
        batch_data = self.conv6(batch_data)
        batch_data = self.maxpool2(batch_data)

        return int(np.prod(batch_data.size()))

    def forward(self, batch_data):
        batch_data = T.tensor(batch_data,dtype=T.float32).to(self.device)
        #print("forward size:{}".format(batch_data.size()))
        #batch_data=batch_data.view(1,1,28,28)
    
        batch_data = self.conv1(batch_data)
        batch_data = self.bn1(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv2(batch_data)
        batch_data = self.bn2(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv3(batch_data)
        batch_data = self.bn3(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.maxpool1(batch_data)

        batch_data = self.conv4(batch_data)
        batch_data = self.bn4(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv5(batch_data)
        batch_data = self.bn5(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv6(batch_data)
        batch_data = self.bn6(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.maxpool2(batch_data)

        batch_data = batch_data.view(batch_data.size()[0], -1)

        classes = self.fc1(batch_data)

        return classes




network = CNN(lr=0.001, batch_size=128, epochs=25)
network.load_state_dict(T.load('./model.tar')['model_state_dict'])
network.optimizer=T.load('./model.tar')['optimizer_state_dict']
network.eval()

Image_Data=get_Image_data()
sudoku=np.zeros((9,9))
device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

print("size of Image_Data:{}".format(len(Image_Data)))
cnt=0
for x,y,img in Image_Data:
    img=img[2:-2,2:-2]
    img=cv2.resize(img,(28,28))
    pred_img=img
    img=[[img] for _ in range(128)]
    img=np.array(img)
    #img.view('float32')
    #print("img:{}".format(img))
    #break
    #print("x:{} y:{},img_shape:{}".format(x,y,np.shape(img)))
    img=T.tensor(img,dtype=T.float32).to(device)

    #print("img dtype:{}".format(img.dtype))
    #break
    #img=[img]
    #img=T.tensor(img).to(device)
    #img=img.view(1,1,28,28)
    prediction=network.forward(img)
    #break
    prediction=F.softmax(prediction)
    print("prediction:{}".format(prediction))
    pred=T.argmax(prediction,dim=1)

    ans=0
    if prediction[0][pred[0]]>0.97:
        ans=pred[0].item()
    print("x:{} ,y:{} ,ans:{},no:{}".format(x,y,ans,cnt))
    print("pred:{}".format(pred[0]))
    cnt+=1
    cv2.imshow('i',pred_img)
    cv2.waitKey(300) #change to your own waiting time 1000 = 1 second
    #if pred<0.7:
    #    pred=0
    
    #x,y=p[0],p[1]
    sudoku[x][y]=ans
cv2.destroyAllWindows()

print(sudoku)

grid=list(sudoku)


def check(x,y,n):
    #rowcheck
    for i in range(9):
        if grid[x][i]==n:
            return False
    for i in range(9):
        if grid[i][y]==n:
            return False
    for i in range((x//3)*3,(x//3)*3+3):
        for j in range((y//3)*3,(y//3)*3+3):
            if grid[i][j]==n:
                return False
    return True

def solve():
    global grid
    #import numpy as np

    for x in range(9):
        for y in range(9):
            if grid[x][y]==0:
                for n in range(1,10):
                    if check(x,y,n):
                        grid[x][y]=n
                        solve()
                        grid[x][y]=0
                return
    print("Solved grid:{}".format(np.matrix(grid)))
    return np.matrix(grid)
Ans=solve()
print("Ans:{}".format(Ans))
#display(grid,points)


