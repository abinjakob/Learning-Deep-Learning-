#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 17:57:40 2023

Basic Python Coding from Deep Learning Tutorial by Mike Cohen 
Section 08 - Cross Validation : Applying cross validation on simple regression

Here the code from the file S08_CrossValidation_onRegression.py is modified to use
scikitlearn and dataloader methods to split the train and test data instead of the 
manual method. 

Also the ratio of split is not hard coded and an additional variable used to define this 

Similar to the previous file the goal is to Here we are creating a model that can predict 
the values of Y based on the values of X


@author: abinjacob
"""

#%% libraries 

import torch 
import torch.nn as nn 
import numpy as np
import matplotlib.pyplot as plt

# for scikitlearn
from sklearn.model_selection import train_test_split

# for DataLoader
from torch.utils.data import DataLoader, TensorDataset

#%% create data
N = 1000
x = torch.randn(N,1)
y = x + torch.randn(N,1)

# plot the data
plt.plot(x,y,'s')

#%% building the model 

ANNreg = nn.Sequential(
    nn.Linear(1,1),             # input layer
    nn.ReLU(),                  # activation layer 
    nn.Linear(1,1)              # output layer
    )

# loss function 
lossfun = nn.MSELoss()

# optimizer 
optimizer = torch.optim.SGD(ANNreg.parameters(), lr = .05)

#%% selecting data for training using scikitlearn

# splitting ratio
trainSplit = [.8,.2]

# splitting the data to train and test sets 
x_trainSL, x_testSL, y_trainSL, y_testSL = train_test_split(x, y, train_size = trainSplit[0])

#%% train the model with scikitlearn split

numepochs = 2000

# loop over epochs
for epochi in range(numepochs):
    
    # forward pass
    yHat = ANNreg(x_trainSL)
    
    # compute loss
    loss = lossfun(yHat, y_trainSL)
    
    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
#%% report losses 

# compute losses of TEST set
predYtest = ANNreg(x_testSL)
testloss = (predYtest - y_testSL).pow(2).mean()

# print final TRAIN loss and TEST loss
print(f'Final Train Loss: {loss.detach():.2f}')
print(f'Final Test Loss: {testloss.detach():.2f}')

#%% plot the data

# predictions for final training run
predYtrain = ANNreg(x_trainSL).detach().numpy()

# now plot
plt.plot(x,y,'k^',label='All data')
plt.plot(x_trainSL, predYtrain,
         'bs',markerfacecolor='w',label='Training pred.')
plt.plot(x_testSL,predYtest.detach(),
         'ro',markerfacecolor='w',label='Test pred.')
plt.legend()






