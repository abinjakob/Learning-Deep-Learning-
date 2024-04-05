#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 15:56:50 2023

Basic Python Coding from Deep Learning Tutorial by Mike Cohen 
Section 07 - Artificial Neural Networks : nn.Sequential vs own Classes

Contains:
    Creating your own classes for creating the models than using nn.Sequential

@author: abinjacob
"""

#%% Libraries 
import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F

import matplotlib.pyplot as plt
from IPython import display
display.set_matplotlib_formats('svg')

#%% create data

nPerClust = 100
blur = 1

A = [  1, 1 ]
B = [  5, 1 ]

# generate data
a = [ A[0]+np.random.randn(nPerClust)*blur , A[1]+np.random.randn(nPerClust)*blur ]
b = [ B[0]+np.random.randn(nPerClust)*blur , B[1]+np.random.randn(nPerClust)*blur ]

# true labels
labels_np = np.vstack((np.zeros((nPerClust,1)),np.ones((nPerClust,1))))

# concatanate into a matrix
data_np = np.hstack((a,b)).T

# convert to a pytorch tensor
data = torch.tensor(data_np).float()
labels = torch.tensor(labels_np).float()

# show the data
fig = plt.figure(figsize=(5,5))
plt.plot(data[np.where(labels==0)[0],0],data[np.where(labels==0)[0],1],'bs')
plt.plot(data[np.where(labels==1)[0],0],data[np.where(labels==1)[0],1],'ko')
plt.title('The qwerties!')
plt.xlabel('qwerty dimension 1')
plt.ylabel('qwerty dimension 2')
plt.show()

#%% Building the model using classes 

# defining the  class
class theClass4ANN(nn.Module):
    
    # creates the layers 
    def __init__(self):
        super().__init__()
        
        # input layer 
        self.input = nn.Linear(2,1)
        
        # output layer 
        self.output = nn.Linear(1,1)
        
    
    # forward pass
    def forward(self,x):
        
        # pass through the input layer 
        x = self.input(x)
        
        # apply relu
        x = F.relu(x)
        
        # output layer 
        x = self.output(x)
        x = torch.sigmoid(x)
        
        return x
    
# creating an instance of the class
ANNClassify = theClass4ANN()            # we need to creat an instance of the class to work with it

#%% other model features
learningRate = .01

# loss function
lossfun = nn.BCELoss()

# optimizer
optimizer = torch.optim.SGD(ANNClassify.parameters(), lr= learningRate)

#%% Train the Model

numepochs = 1000
losses = torch.zeros(numepochs)

for epochi in range(numepochs):
    
    # forward pass
    yHat = ANNClassify(data)
    
    # compute loss
    loss = lossfun(yHat, labels)
    losses[epochi] = loss
    
    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#%% show the losses

plt.plot(losses.detach(),'o', markerfacecolor= 'w', linewidth= .1)   

#%% compute the predictions

# manually compute losses
# final forward pass
predictions = ANNClassify(data)

predlabels = predictions>.5

# find errors
misclassified = np.where(predlabels != labels)[0]

# total accuracy
totalacc = 100-100*len(misclassified)/(2*nPerClust)

print('Final accuracy: %g%%' %totalacc)    
    
#%% plot the labeled data
fig = plt.figure(figsize=(5,5))
plt.plot(data[misclassified,0] ,data[misclassified,1],'rx',markersize=12,markeredgewidth=3)
plt.plot(data[np.where(~predlabels)[0],0],data[np.where(~predlabels)[0],1],'bs')
plt.plot(data[np.where(predlabels)[0],0] ,data[np.where(predlabels)[0],1] ,'ko')

plt.legend(['Misclassified','blue','black'],bbox_to_anchor=(1,1))
plt.title(f'{totalacc}% correct')
plt.show()
    
    
    
    
    
    









        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    



















