## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.batchn1 = nn.BatchNorm2d(32) 

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
         # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        # second conv layer: 10 inputs, 20 outputs, 5x5 conv
        ## output size = (W-F)/S +1 = (13-5)/1 +1 = 9
        # the output tensor will have dimensions: (20, 9, 9)
        # after another pool layer this becomes (20, 4, 4); 4.5 is rounded down
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.batchn2 = nn.BatchNorm2d(64) 

        self.conv3 = nn.Conv2d(64, 128, 5)
        self.batchn3 = nn.BatchNorm2d(128) 

        self.conv4 = nn.Conv2d(128, 256, 5)
        self.batchn4 = nn.BatchNorm2d(256) 


        # 20 outputs * the 4*4 filtered/pooled map size
        self.fc1 = nn.Linear(256*10*10, 1024)
        
        # dropout with p=0.5
        self.fc1_drop = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(1024, 136)


        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.batchn1(self.conv1(x))))
        
        x = self.pool(F.relu(self.batchn2(self.conv2(x))))
        
        x = self.pool(F.relu(self.batchn3(self.conv3(x))))
        
        x = self.pool(F.relu(self.batchn4(self.conv4(x))))
       

        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        
        # print("conv1", x.size())
        # print("conv2", x.size())
        # print("conv3", x.size())
        # print("conv4", x.size())
        # print("flatten", x.size())

        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
