#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 16:23:54 2023

Basic Python Coding from Deep Learning Tutorial by Mike Cohen 
Section 07 - Artificial Neural Networks : Multi Output ANN
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

import seaborn as sns

#%% import dataset

iris = sns.load_dataset('iris')         # this dataset comes with seaborn

# checking first few lines of data
iris.head()

# plotting the data 
sns.pairplot(iris, hue= 'species')

#%% organise the data

# converting from pandas to tensor 
data = torch.tensor(iris[iris.columns[0:4]].values).float()

# transform species label to number
labels = torch.zeros(len(data), dtype= torch.long)
# labels: setosa=0, versicolor=1, virginica=2
# as labels is initialsed as 0 no need to explicitly set setosa
labels[iris.species == 'versicolor'] = 1
labels[iris.species == 'virginica'] = 2

#%% create ANN Model
# check notes for the model architecture design

learningRate = .01

# model architecture
ANNiris = nn.Sequential(
    nn.Linear(4,64),               # input layer 
    nn.ReLU(),                     # activation
    nn.Linear(64,64),              # hidden layer
    nn.ReLU(),                     # activation
    nn.Linear(64,3),               # output layer     
    )

# loss function
lossfun = nn.CrossEntropyLoss()    # this combines the Softmax function and the loss function

# optimizer
optimizer = torch.optim.SGD(ANNiris.parameters(), lr= learningRate)

#%% Training

numepochs = 1000

# initialize losses
losses = torch.zeros(numepochs)
ongoingAcc = []

# training 
for epochi in range(numepochs):
    
    # forward pass
    yHat = ANNiris(data)
    
    # compute loss
    loss = lossfun(yHat, labels)
    losses[epochi] = loss
    
    #backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # training loop complete 
    
    # copute accuracy at each epochs 
    matches = torch.argmax(yHat, axis=1) == labels          # booleans (true/false)
    matchesNumeric = matches.float()                        # convert to numbers (0/1)
    accuracyPct = 100*torch.mean(matchesNumeric)            # average and percentage 
    ongoingAcc.append(accuracyPct)                          # add to list of accuracies
    
# final forward pass
predictions = ANNiris(data)

predlabels = torch.argmax(predictions, axis=1)
totalacc = 100*torch.mean((predlabels == labels).float())

#%% Visualise results

# report accuracy
print(f'Final accuracy: {totalacc}%')

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
# run training again to see whether this performance is consistent























