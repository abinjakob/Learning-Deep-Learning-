#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 19:24:55 2023

Basic Python Coding from Deep Learning Tutorial by Mike Cohen 
Section 07 - Artificial Neural Networks Part 02 Binary Classification: Learning rate comparison 

@author: abinjacob
"""

#%% Libraries 

import torch 
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
import matplotlib.pyplot as plt

#%%  create data

nPerClust = 100
blur = 1

A = [1,3]
B = [1,-2]

# generate data
a = [ A[0]+np.random.randn(nPerClust)*blur , A[1]+np.random.randn(nPerClust)*blur ]
b = [ B[0]+np.random.randn(nPerClust)*blur , B[1]+np.random.randn(nPerClust)*blur ]

# true labels
labels_np = np.vstack((np.zeros((nPerClust,1)), np.ones((nPerClust,1))))

# concatenate into a matrix
data_np  = np.hstack((a,b)).T

# convert to pytorch tensor
data = torch.tensor(data_np).float()
labels = torch.tensor(labels_np).float()

# show the data
fig = plt.figure(figsize= (5,5))
plt.plot(data[np.where(labels== 0)[0],0], data[np.where(labels== 0)[0],1],'bs')
plt.plot(data[np.where(labels== 1)[0],0], data[np.where(labels== 1)[0],1],'ko')
plt.title('The qwerties')
plt.xlabel('qwerty dimension 1')
plt.xlabel('qwerty dimension 2')

#%% function to build the model

# def createANNmodel(learningRate):
    
#     # model architecture
#     ANNClassify = nn.Sequential(
#         nn.Linear(2, 16),                # input layer               | 2 inputs and 16 outputs from this node
#         nn.ReLU(),                       # activation unit
#         nn.Linear(16,1),                 # hiddeb layer              | 16 inputs and 1 output from this node
#         nn.ReLU(),                       # activation unit
#         nn.Linear(1, 1),                 # output unit               | 1 input and 1 output from the node
#         nn.Sigmoid(),                    # final activation unit               
#         )
    
#     # loss function
#     lossfun = nn.BCELoss()
    
#     # optimizer
#     optimizer = torch.optim.SGD(ANNClassify.parameters(), lr= learningRate)
    
    
#     # model output
#     return ANNClassify, lossfun, optimizer


#%% -> Converting the above nn.Sequential to a class

# function to build the model 
def createANNmodel(learningRate): 
    
    # create a class for the model
    class createANNClassify(nn.Module):
        
        # craete the layers 
        def __init__(self):
            super().__init__()
            
            # input layer 
            self.input = nn.Linear(2, 16)
            
            # hidden layer
            self.hidden = nn.Linear(16, 1)
            
            # output layer 
            self.output = nn.Linear(1, 1)
                
            
        # forward pass
        def forward(self,x):
            
            # pass through input layer 
            x = self.input(x)
            x = F.relu(x)                     # apply relu activation function between input and hidden layer 
            
            # pass thorugh hidden layer 
            x = self.hidden(x)
            x = F.relu(x)                     # apply relu activation function between hidden and output layer 
            
            # pass though output layer 
            x = self.output(x)
            x = torch.sigmoid(x)              # apply sigmoid to final output 
            
            return x 
    
    # create the model instance
    ANNClassify = createANNClassify()
    
    # loss function
    lossfun = nn.BCELoss()
        
    # optimizer
    optimizer = torch.optim.SGD(ANNClassify.parameters(), lr= learningRate)
        
    # model output
    return ANNClassify, lossfun, optimizer
    


#%% function to train the model

# fixed parameters 
numepochs = 1000

def trainTheModel(ANNmodel):
    
    # intialise losses
    losses = torch.zeros(numepochs)
    
    
    # running the model
    for epochi in range(numepochs):
        
        # forward pass
        yHat = ANNmodel(data)
        
        #compute loss
        loss = lossfun(yHat,labels)
        losses[epochi] = loss
        
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # end of training 
        
    # final forward pass
    predictions = ANNmodel(data)
    
    # compute the predictions and report accuracy
    totalacc = 100*torch.mean(((predictions>.5) == labels).float())   # checking greater than .5 as the final output is from sigmoid 
                                                                      #(sigmoid wriiten explicitly)
    
    # output
    return losses, predictions, totalacc

#%% testing the code for errors

# create
ANNClassify, lossfun, optimizer = createANNmodel(.01)

# running 
losses, predictions, totalacc = trainTheModel(ANNClassify)

# reporting accuracy
print(totalacc)

# plotting the losses
plt.plot(losses.detach(),'o', markerfacecolor= 'w', linewidth= .1)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'Accuracy = {totalacc}')

 
    
    
    
    
    
    
    
    
    
