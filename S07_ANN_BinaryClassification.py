#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 17:30:13 2023

Basic Python Coding from Deep Learning Tutorial by Mike Cohen 
Section 07 - Artificial Neural Networks Part 02 Binary Classification
Contains:

@author: abinjacob
"""

#%% Libraries 

import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt

from IPython import display
display.set_matplotlib_formats('svg')

#%% create data

nPerClust = 100
blur = 1

A = [ 1,1 ]
B = [ 5,1 ]

# generate data
a = [ A[0]+np.random.randn(nPerClust)*blur , A[1]+np.random.randn(nPerClust)*blur ]
b = [ B[0]+np.random.randn(nPerClust)*blur , B[1]+np.random.randn(nPerClust)*blur ]


# true labels 
labels_np = np.vstack((np.zeros((nPerClust,1)), np.ones((nPerClust,1))))

# concatenate the data into a matrix
data_np = np.hstack((a,b)).T

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


#%% build the model

ANNclassify = nn.Sequential(
    nn.Linear(2,1),                 # input layer. Since we have two inputs and 1 output we write (2,1)
    nn.ReLU(),                      # activation unit
    nn.Linear(1,1),                 # output unit
    nn.Sigmoid())                   # final activation unit

ANNclassify

#%% other model features

learningRate = .01

# loss function
lossfun = nn.BCELoss()              # using binary cross entropy problem

# optimizer
optimizer = torch.optim.SGD(ANNclassify.parameters(),lr=learningRate) 

#%% trainning the model
numepochs = 1000
losses = torch.zeros(numepochs)


# training 
for epochi in range(numepochs):
    
    # forward pass
    yHat = ANNclassify(data)
    
    # compute loss
    loss = lossfun(yHat,labels)
    losses[epochi] = loss
    
    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# training ends 

#%% show the losses

plt.plot(losses.detach(),'o',markerfacecolor= 'w', linewidth= .1)
plt.xlabel('Epoch')
plt.ylabel('Loss')

#%% compute the predictions 

# manually computing losses
# final forward pass
predictions = ANNclassify(data)         # the final output is from sigmoid so it is in range 0 to 1
predlabels = predictions>.5             # just binarising this. If >.5 True else False

# find errors
misclassified = np.where(predlabels != labels)[0]

# total accuracy
totalacc = 100-100*len(misclassified)/(2*nPerClust)

print(f'Final Accuracy:  {totalacc}')


#%% plot the labelled data

fig = plt.figure(figsize=(5,5))
plt.plot(data[misclassified,0] ,data[misclassified,1],'rx',markersize=12,markeredgewidth=3)
plt.plot(data[np.where(~predlabels)[0],0],data[np.where(~predlabels)[0],1],'bs')
plt.plot(data[np.where(predlabels)[0],0] ,data[np.where(predlabels)[0],1] ,'ko')

plt.legend(['Misclassified','blue','black'],bbox_to_anchor=(1,1))
plt.title(f'{totalacc}% correct')
plt.show()








    
    
    
    
    
    






















