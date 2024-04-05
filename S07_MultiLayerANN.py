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

#%% fundtion to build the model

def createANNmodel(learningRate):
    
    # model architecture
    ANNClassify = nn.Sequential(
        nn.Linear(2, 16),                # input layer               | 2 inputs and 16 outputs from this node
        nn.ReLU(),                       # activation unit
        nn.Linear(16,1),                 # hiddeb layer              | 16 inputs and 1 output from this node
        nn.ReLU(),                       # activation unit
        nn.Linear(1, 1),                 # output unit               | 1 input and 1 output from the node
        nn.Sigmoid(),                    # final activation unit               
        )
    
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

        
#%% test with varying learning rates

learningrates =  np.linspace(.001, .1, 50)

# initialise 
accByLR = []
allLosses = np.zeros((len(learningrates), numepochs))

# looping over  learning rates
for i,lr in enumerate(learningrates):
    
    # create and run the model
    ANNClassify, lossfun, optimizer = createANNmodel(lr)
    losses, predictions, totalacc = trainTheModel(ANNClassify)
    
    # store the results
    accByLR.append(totalacc)
    allLosses[i,:] =  losses.detach()
    
#%% plot the results


fig,ax = plt.subplots(1,2,figsize=(16,4))

ax[0].plot(learningrates,accByLR,'s-')
ax[0].set_xlabel('Learning rate')
ax[0].set_ylabel('Accuracy')
ax[0].set_title('Accuracy by learning rate')

ax[1].plot(allLosses.T)
ax[1].set_title('Losses by learning rate')
ax[1].set_xlabel('Epoch number')
ax[1].set_ylabel('Loss')
plt.show()

# proportion of runs where the model had at least 70% accuracy
sum(torch.tensor(accByLR)>70)/len(accByLR)

    
    
    
    
    
    
    
    
    
    
