#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 17:16:51 2023

Basic Python Coding from Deep Learning Tutorial by Mike Cohen 
Section 07 - Artificial Neural Networks : Code Challenge 

Integrate S07_ANN_BinaryClassification.py and S07_Multi-OutputANN.py 
Make three groups of qwerties and train a 3-output ANN to classify them

@author: abinjacob
"""

#%% Libraries

import torch 
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from IPython import display
display.set_matplotlib_formats('svg')


#%% Creating the data

nPerClust = 100
blur = 1

# just create the positions of where the data would be in the graph
A = [1,1]
B = [5,1]
C = [3,-2]

# generate the data 
a = [ A[0]+np.random.randn(nPerClust)*blur , A[1]+np.random.randn(nPerClust)*blur ]
b = [ B[0]+np.random.randn(nPerClust)*blur , B[1]+np.random.randn(nPerClust)*blur ]
c = [ C[0]+np.random.randn(nPerClust)*blur , C[1]+np.random.randn(nPerClust)*blur ]

# true labels 
labels_np = np.vstack( (np.full((nPerClust,1),0), np.full((nPerClust,1),1), np.full((nPerClust,1),2)) )

# concatenate into matrix
data_np = np.hstack((a,b,c)).T


# convert to PyTorch 
data = torch.tensor(data_np).float()
labels = torch.squeeze(torch.tensor(labels_np).long())

# show the data
fig = plt.figure(figsize=(5,5))
plt.plot(data[np.where(labels==0)[0],0],data[np.where(labels==0)[0],1],'bs')
plt.plot(data[np.where(labels==1)[0],0],data[np.where(labels==1)[0],1],'ko')
plt.plot(data[np.where(labels==2)[0],0],data[np.where(labels==2)[0],1],'r^')
plt.title('The qwerties!')
plt.xlabel('qwerty dimension 1')
plt.ylabel('qwerty dimension 2')
plt.show()

#%% build the model
# modelarchitecture is 2-4-3

learningRate = .01

# model architecture
ANNQwerty = nn.Sequential(
    nn.Linear(2,4),               # input layer 
    nn.ReLU(),                    # activation
    nn.Linear(4,3),               # hidden layer
    nn.Softmax(dim= 1),           # activation    
    )

# loss function
lossfun = nn.CrossEntropyLoss()    # this combines the Softmax function and the loss function

# optimizer
optimizer = torch.optim.SGD(ANNQwerty.parameters(), lr= learningRate)


#%% training

numepochs = 10000

#initialise losses
losses = torch.zeros(numepochs)
ongoingAcc = []

# trainig the model
for epochi in range(numepochs):
    
    # forward pass
    yHat = ANNQwerty(data)
    
    # compute loss
    loss = lossfun(yHat,labels)
    losses[epochi] = loss
    
    #backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # end of training loop
    
    
    # compute accuracy
    matches = torch.argmax(yHat,axis=1) == labels     # booleans (false/true)
    matchesNumeric = matches.float()                  # convert to numbers (0/1)
    accuracyPct = 100*torch.mean(matchesNumeric)      # average and x100 
    ongoingAcc.append( accuracyPct )                  # add to list of accuracies
    
# final forward pass
predictions = ANNQwerty(data)

predlabels = torch.argmax(predictions,axis=1)
totalacc = 100*torch.mean((predlabels == labels).float())

#%% visualise the results

# report accuracy
print('Final accuracy: %g%%' %totalacc)

fig,ax = plt.subplots(1,2,figsize=(13,4))

ax[0].plot(losses.detach())
ax[0].set_ylabel('Loss')
ax[0].set_xlabel('epoch')
ax[0].set_title('Losses')

ax[1].plot(ongoingAcc)
ax[1].set_ylabel('accuracy')
ax[1].set_xlabel('epoch')
ax[1].set_title('Accuracy')
plt.show() 


# plot the raw model outputs

fig = plt.figure(figsize=(10,4))

colorshape = [  'bs','ko','r^' ]
for i in range(3):
  plt.plot(yHat[:,i].detach(),colorshape[i],markerfacecolor='w')

plt.xlabel('Stimulus number')
plt.ylabel('Probability')
plt.legend(['qwert 1','qwert 2','qwert 3'],loc=(1.01,.4))
plt.show()


