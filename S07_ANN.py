#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 17:03:48 2023

Basic Python Coding from Deep Learning Tutorial by Mike Cohen 
Section 07 - Artificial Neural Networks 
Contains:


@author: abinjacob
"""

#%% Libraries 

import numpy as np
import torch
import torch.nn as nn 
import matplotlib.pyplot as plt 

from IPython import display
display.set_matplotlib_formats('svg')


#%% Create Data  

N = 30
x = torch.randn(N,1)
y = x + torch.randn(N,1)/2

# plotting
plt.plot(x,y,'s')

# building the  model
ANNreg = nn.Sequential(
    nn.Linear(1,1),      # input layer > The number 1,1 means the model take 1 input and gives 1 output (given to output layer through ReLU)
    nn.ReLU(),           # activation function (ReLU- Rectified Linear Unit)
    nn.Linear(1,1)       # output layer 
    )

ANNreg


#%% Set Parameters 
    
#learning rate
learningRate = .05

# loss function using MSE as it is a cont. prediction of numerical values
lossfun = nn.MSELoss()  

# optimizer (the type of gradient descent we are going to be using)
optimizer = torch.optim.SGD(ANNreg.parameters(),lr=learningRate)


#%% Train the Model

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

    
#%% Just manually computing the losses

# final forward pass
predictions = ANNreg(x)

# final loss (MSE)
testloss = (predictions-y).pow(2).mean()


plt.figure()
plt.plot(losses.detach(),'o',markerfacecolor='w',linewidth=.1)
plt.plot(numepochs,testloss.detach(),'ro')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Final loss = %g' %testloss.item())
plt.show()    

# plotting the data 
plt.figure()
plt.plot(x,y,'bo',label='Real data')
plt.plot(x,predictions.detach(),'rs',label='Predictions')
plt.title(f'prediction-data r={np.corrcoef(y.T,predictions.detach().T)[0,1]:.2f}')
plt.legend()
plt.show()
































