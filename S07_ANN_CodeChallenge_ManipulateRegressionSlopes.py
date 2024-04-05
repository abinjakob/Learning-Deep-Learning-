#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 17:12:13 2023

Basic Python Coding from Deep Learning Tutorial by Mike Cohen 
Section 07 - Artificial Neural Networks 
Code Challenge: Manipulate regression slopes
- created function that outputs x and y data with a given slope
- created a function for building and training the model
- visualisation of the model performance

@author: abinjacob
"""

#%% Libraries

import numpy as np
import torch
import torch.nn as nn 
import matplotlib.pyplot as plt 

from IPython import display
display.set_matplotlib_formats('svg')


#%% function to create the data
def regData(m):
    N = 50
    x = torch.randn(N,1)
    y = m*x + torch.randn(N,1)/2
    
    return x,y


#%% function for training 

def regModel(x,y):
    
    # building the model
    ANNreg = nn.Sequential(
        nn.Linear(1,1),      # input layer > The number 1,1 means the model take 1 input and gives 1 output (given to output layer through ReLU)
        nn.ReLU(),           # activation function (ReLU- Rectified Linear Unit)
        nn.Linear(1,1)       # output layer 
        ) 
    
    # setting training paramenters 
    #learning rate
    learningRate = .05

    # loss function using MSE as it is a cont. prediction of numerical values
    lossfun = nn.MSELoss()  

    # optimizer (the type of gradient descent we are going to be using)
    optimizer = torch.optim.SGD(ANNreg.parameters(),lr=learningRate)
    
    
    # training the model
    numepochs  = 500
    losses = torch.zeros(numepochs)




    # Training Model
    for epochi in range(numepochs):
        
        # forward pass
        yHat = ANNreg(x)
        
        # compute loss
        loss = lossfun(yHat,y)
        losses[epochi] = loss               # just storing the losses to visualise it 
        
        #backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # end of training loop
    
    
    
    # compute model prediction
    predictions = ANNreg(x)

    # Output:
    return predictions, losses


#%% Just testing if everything works (Often a good practice to test everything before going to the long process)

# create data
x,y = regData(.8)

# run model
yHat, losses = regModel(x, y)

# plotting 
fig,ax = plt.subplots(1,2,figsize=(12,4))

ax[0].plot(losses.detach(),'o',markerfacecolor='w',linewidth=.1)
ax[0].set_xlabel('Epoch')
ax[0].set_title('Loss')

ax[1].plot(x,y,'bo',label='Real data')
ax[1].plot(x,yHat.detach(),'rs',label='Predictions')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[1].legend()
ax[1].set_title(f'prediction-data corr = {np.corrcoef(y.T,yHat.detach().T)[0,1]:.2f}')


    
#%% parametric experiment - vary slope from -2 to 2 in 21 steps

# the slopes
m = np.linspace(-2,2,21)

# number of trials 
trials = 50

# initialise an output matrix 
results = np.zeros((len(m),trials,2))

for mi in range(len(m)):                            # looping for slopes 
    for trialNum in range(trials):                  # each slopes will run for 50 trials
        
        # create the data
        x,y = regData(m[mi])
        
        # run the model
        yHat, losses = regModel(x, y)
        
        # store the final loss and performance 
        results[mi,trialNum,0] = losses[-1]
        results[mi,trialNum,1] = np.corrcoef(y.T,yHat.detach().T)[0,1]
        
# correlation can be 0 if the model didn't do well. Set nan's->0
results[np.isnan(results)] = 0


#%% plotting the results

fig,ax = plt.subplots(1,2,figsize=(12,4))

ax[0].plot(m,np.mean(results[:,:,0],axis=1),'ko-',markerfacecolor='w',markersize=10)
ax[0].set_xlabel('Slope')
ax[0].set_title('Loss')

ax[1].plot(m,np.mean(results[:,:,1],axis=1),'ms-',markerfacecolor='w',markersize=10)
ax[1].set_xlabel('Slope')
ax[1].set_ylabel('Real-predicted correlation')
ax[1].set_title('Model Performance')

























