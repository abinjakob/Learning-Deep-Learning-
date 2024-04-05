#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 14:03:34 2023

Basic Python Coding from Deep Learning Tutorial by Mike Cohen 
Section 08 - Overfitting and Cross Validation : Manual Separation

Manually applying cross-validation using numpy

Here we devide:
    80% of the data into a Training Set
    10% into a dev set / hold test  -> not used for this program
    10% into a test set

@author: abinjacob
"""

#%% Libraries

import torch 
import torch.nn as nn
import numpy as np

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

#%% separating data into train and test
# (no devset used)

# define training examples
propTraining = .8 # using  proportion instead of percentage
nTraining = int(len(labels)*propTraining)

# # initialise a boolean vector to select data and labels
trainTestBool = np.zeros(len(labels), dtype= bool)

# # set the 80% if values as true
# trainTestBool[range(nTraining)] = True 

itmes2use4train = np.random.choice(range(len(labels)),nTraining,replace= False)
trainTestBool[itmes2use4train]  = True

#%% create ANN model

# model architecture 
ANNiris = nn.Sequential(
    nn.Linear(4, 64),           # input layer
    nn.ReLU(),                  # activation unit
    nn.Linear(64, 64),          # hidden layer 
    nn.ReLU(),                  # activation unit
    nn.Linear(64, 3),           # output unit
    )

# loss function 
lossfun = nn.CrossEntropyLoss()

# optimizer 
optimizer = torch.optim.SGD(ANNiris.parameters(), lr= .01)

#%% just looking at the data and training set

# entire data 
print(data.shape)

# training set
print(data[trainTestBool,:].shape)

# test set
print(data[~trainTestBool,:].shape)

#%% training the model

numepochs = 1000

# initialise losses 
losses = torch.zeros(numepochs)
ongoingAcc = []

# loop over epochs 
for epochi in range(numepochs):
    
    # forward pass
    yHat = ANNiris(data[trainTestBool,:])  # note that here we are using only the 80% of data
    
    # compute accuracy
    ongoingAcc.append(100 * torch.mean((torch.argmax(yHat, axis= 1) == labels[trainTestBool]).float() ))
    
    
    # compute loss
    loss = lossfun(yHat, labels[trainTestBool])
    losses[epochi] = loss
    
    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#%% compute accuracy for train and test

# final forward pass using Test Data
predictions = ANNiris(data[trainTestBool,:])
trainacc = 100 * torch.mean((torch.argmax(predictions, axis= 1) == labels[trainTestBool]).float())

# final forward pass using Test Data
predictions = ANNiris(data[~trainTestBool,:])
testacc = 100 * torch.mean((torch.argmax(predictions, axis= 1) == labels[~trainTestBool]).float())

# report accuracy
print(f'Final Train Accuracy: {trainacc}')
print(f'Final Test Accuracy: {testacc}')




