#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 21:50:57 2023

Basic Python Coding from Deep Learning Tutorial by Mike Cohen 
Section 08 - Cross Validation : Applying cross validation on simple regression

Here we are creating a model that can predict the values of Y based on the values of X

@author: abinjacob
"""

#%% libraries 
import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

#%% create data
N = 100
x = torch.randn(N,1)
y = x + torch.randn(N,1)

# plot the data
plt.plot(x,y,'s')

#%% building the model

ANNreg = nn.Sequential(
    nn.Linear(1, 1),        # input layer 
    nn.ReLU(),              # activation function
    nn.Linear(1, 1)         # output layer    
    )

#%% model meta-parameters

learningRate = .05

# loss function
lossfun = nn.MSELoss()

# optimizer 
optimizer = torch.optim.SGD(ANNreg.parameters(), lr= learningRate)

#%% select data for training 
# note: the data is selected by creating a variable with bools of True and False
# and then x[True] will be selected

# random indices for True
trainidx = np.random.choice(range(N), 80, replace= False)  # here the split is hard coded to 80%
# initialising a vector with all Falses 
trainBool = np.zeros(N, dtype= bool)
# set selected samples to true
trainBool[trainidx] = True

# show the sizes
print(x[trainBool].shape)       # 80 random values of x is selected for training set (where trainBool= True)
print(x[~trainBool].shape)      # 20 random values of x is selected test set (where trainBool= False)

#%% train the model

numepochs = 500

# loop over epochs 
for epochi in range(numepochs):
     
    # forward pass
    yHat = ANNreg(x[trainBool])
    
    # compute loss
    loss = lossfun(yHat, y[trainBool])
    
    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
#%% report losses 

# compute losses of TEST set
predYtest = ANNreg(x[~trainBool])
testloss = (predYtest - y[~trainBool]).pow(2).mean()

# print final TRAIN loss and TEST loss
print(f'Final Train Loss: {loss.detach():.2f}')
print(f'Final Test Loss: {testloss.detach():.2f}')

#%% plot the data

# predictions for final training run
predYtrain = ANNreg(x[trainBool]).detach().numpy()

# now plot
plt.plot(x,y,'k^',label='All data')
plt.plot(x[trainBool], predYtrain,
         'bs',markerfacecolor='w',label='Training pred.')
plt.plot(x[~trainBool],predYtest.detach(),
         'ro',markerfacecolor='w',label='Test pred.')
plt.legend()



