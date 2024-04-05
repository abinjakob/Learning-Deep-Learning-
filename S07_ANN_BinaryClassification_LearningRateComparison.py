#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 17:30:13 2023

Basic Python Coding from Deep Learning Tutorial by Mike Cohen 
Section 07 - Artificial Neural Networks Part 02 Binary Classification: Learning rate comparison 
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


#%% Function for building the model

def createANNmodel(learningRate):
    
    # model architecture

    ANNclassify = nn.Sequential(
        nn.Linear(2,1),                 # input layer. Since we have two inputs and 1 output we write (2,1)
        nn.ReLU(),                      # activation unit
        nn.Linear(1,1),                 # output unit
        # nn.Sigmoid())                 # final activation unit (not used because we use BCEWithLogitsLoss() 
        )                                # which implement sigmoid more efficiently internally - Recommended 
                                        # by PyTorch)

    # loss function
    lossfun = nn.BCEWithLogitsLoss()

    # optimizer
    optimizer = torch.optim.SGD(ANNclassify.parameters(),lr=learningRate) 
    
    # model output
    return ANNclassify, lossfun, optimizer

#%% function to train the model

# fixed parameter
numepochs = 1000


def trainTheModel(ANNmodel):
    
    # initialise losses
    losses = torch.zeros(numepochs)
    
    # training 
    for epochi in range(numepochs):
        
        # forward pass
        yHat = ANNmodel(data)
        
        # compute loss
        loss = lossfun(yHat,labels)
        losses[epochi] = loss
        
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # training ends 
    
    # final forward pass
    predictions = ANNmodel(data)
    
    # compute the predictions and report accuracy
    totalacc = 100*torch.mean(((predictions>0) == labels).float())
    # here we are checking if predictions >0 and not 0.5 because here we are sending it through
    # BCEWithLogitsLoss() and not explicitly using a sigmoid() as in the previous example.
    
    # output
    return losses, predictions, totalacc
    

#%% Just testign the code before performing parameter tests

# # Create 
# ANNclassify, lossfun, optimizer = createANNmodel(.01)

# # run it
# losses, predictionsm, totalacc = trainTheModel(ANNclassify)

# # report accuracy
# print(totalacc)

# # show the losses
# plt.plot(losses.detach(), 'o', markerfacecolor= 'w', linewidth= .1)
# plt.xlabel('Epoch')
# pl.ylabel('Loss')


#%% experiment: testing with different learning rates

learningrates = np.linspace(.001,.1,40)

# initialize results output
accByLR = []
allLosses = np.zeros((len(learningrates), numepochs))

# looping through learning rates 
for i,lr in enumerate(learningrates):
    
    # create the model
    ANNclassify, lossfun, optimizer = createANNmodel(lr)
    
    # run the training
    losses, predictions, totalacc = trainTheModel(ANNclassify)
    
    # store the output
    accByLR.append(totalacc)
    allLosses[i,:] = losses.detach()
    
#%% plotting the results

fig,ax = plt.subplots(1,2,figsize=(12,4))

ax[0].plot(learningrates,accByLR,'s-')
ax[0].set_xlabel('Learning rate')
ax[0].set_ylabel('Accuracy')
ax[0].set_title('Accuracy by learning rate')

ax[1].plot(allLosses.T)
ax[1].set_title('Losses by learning rate')
ax[1].set_xlabel('Epoch number')
ax[1].set_ylabel('Loss')
plt.show()

#%% also another way to check the accuracy

# proportion of runs where the model had at least 70% accuracy
sum(torch.tensor(accByLR)>70)/len(accByLR)

# here by looking at the result either the model got good prediction or it was just chance (close to 50%)
# this means chance played a large role in the conclusions 

#%% running the experiment 50 times and average the results: meta-experiment 

# number of trials 
numExps = 50
learningrates = np.linspace(.001,.1,40)

# matrix to store the results
accMeta =  np.zeros((numExps, len(learningrates)))

# fewer epochs to reduce computation time 
numepochs = 500

# experiment 
for expi in range(numExps):
    for i,lr in enumerate(learningrates):
        
        #  building and running the model
        ANNclassify, lossfun, optimizer = createANNmodel(lr)
        losses, predictions, totalacc = trainTheModel(ANNclassify)
        
        # store the results
        accMeta[expi,i] = totalacc
        
# plotting the results averaged over experiments
plt.plot(learningrates, np.mean(accMeta, axis= 0),'s-')
plt.xlabel('Learning Rates')
plt.ylabel('Accuracy')









    
    
    
    
    
    






















