#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 18:30:41 2023

Basic Python Coding from Deep Learning Tutorial by Mike Cohen 
Section 07 - Artificial Neural Networks : Depth vs Breadth

This is a parametric experiment where we run the model with different widths and depths 

@author: abinjacob
"""

#%% Libraries 

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from IPython import display
display.set_matplotlib_formats('svg')

import seaborn as sns

#%% import and organise data

iris = sns.load_dataset('iris')         # this dataset comes with seaborn

# converting from pandas to tensor 
data = torch.tensor(iris[iris.columns[0:4]].values).float()

# transform species label to number
labels = torch.zeros(len(data), dtype= torch.long)
# labels: setosa=0, versicolor=1, virginica=2
# as labels is initialsed as 0 no need to explicitly set setosa
labels[iris.species == 'versicolor'] = 1
labels[iris.species == 'virginica'] = 2

#%% create a class for the model
# here we need to have variables to iterate the widths and depths 
# nUnits and nLayers 

class ANNiris(nn.Module):
    
    # create the layers 
    def __init__(self,nUnits,nLayers):
        super().__init__()
        
        # create a dictionary to store the layers
        self.layers = nn.ModuleDict()                       # dictionary in nn that is specifically  designed to store the layers of a network
        self.nLayers = nLayers
        
        
        # input layer 
        self.layers['input'] = nn.Linear(4,nUnits)
        
        # hidden layer 
        for i in range(nLayers):
            self.layers[f'hidden{i}'] = nn.Linear(nUnits,nUnits)
            
        # output layer 
        self.layers['output'] = nn.Linear(nUnits,3)
        
        
        
    # forward pass
    def forward(self,x):
        
        # input layer  
        x = self.layers['input'](x)
        
        # hidden layers 
        for i in range(self.nLayers):
            x = F.relu( self.layers[f'hidden{i}'](x) )
            
        # output layer  
        x = self.layers['output'](x)
        
        return x
        
#%% genearate an instance of the model for checking 

nUnitsPerLayer = 12
nLayers = 4
net = ANNiris(nUnitsPerLayer, nLayers)

net

#%% A quick test by running  some numbers through the model
# this ensures that the architecture in internally consistent  

# when you create a model always TEST TEST TEST

# 10 samples, 4 dimensions 
tmpx = torch.randn(10,4)

# run it through DL
y = net(tmpx)

# exam the shape of the output
print(y.shape), print()

# print output
print(y)



#%% a function to train the model

def trainTheModel(theModel):
    
    # define the loss function and optiizer
    lossfun = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(theModel.parameters(), lr = 0.1)
    
    # loop over epochs 
    for epochi in range(numepochs):
        
        # forward pass
        yHat = theModel(data)
        
        # calculate loss
        loss = lossfun(yHat,labels)
        
        #backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # final forward pass to get accuracy
    predictions = theModel(data)
    predlabels = torch.argmax(predictions, axis= 1)        
    acc = 100 * torch.mean((predlabels == labels).float())
    
    # total number of trainable parameters in the model
    nParams = 0
    
    for p in theModel.parameters():
        if p.requires_grad:
            nParams += p.numel()
            
    return acc,nParams
            
            
#%% test the function once

numepochs = 2500
acc = trainTheModel(net)  
        
# check the output
acc                                   # a tuple with accuracy and nparams


#%% doing the  experiment 

# define model parameters
numunits  = np.arange(4,101,3)        # units per hidden layer
numlayers = range(1,6)                # number of hidden layer


# initialise output matrices
accuracies  = np.zeros((len(numunits),len(numlayers)))
totalparams = np.zeros((len(numunits),len(numlayers)))

# number of training epochs 
numepochs = 500


# start the experiment 
for unitidx in range(len(numunits)):
    for layeridx in range(len(numlayers)):
        
        # create  a fresh model instance
        net = ANNiris(numunits[unitidx], numlayers[layeridx])
        
        #  run the model and store the results
        acc,nParams =  trainTheModel(net)
        accuracies[unitidx,layeridx]  = acc
        totalparams[unitidx,layeridx] = nParams

#%% # show accuracy as a function of model depth
        
# show accuracy as a function of model depth
fig,ax = plt.subplots(1,figsize=(12,6))

ax.plot(numunits,accuracies,'o-',markerfacecolor='w',markersize=9)
ax.plot(numunits[[0,-1]],[33,33],'--',color=[.8,.8,.8])
ax.plot(numunits[[0,-1]],[67,67],'--',color=[.8,.8,.8])
ax.legend(numlayers)
ax.set_ylabel('accuracy')
ax.set_xlabel('Number of hidden units')
ax.set_title('Accuracy')
plt.show()        
        
        
        
        


        
        
        
        
        
        
        
        
        




























        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        