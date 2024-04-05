#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 18:47:06 2023

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
print(iris.head())

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

#%% Funcrtion to create ANN Model
# check notes for the model architecture design

def createIrisModel(nHidden):
    
    learningRate = .01

    # model architecture
    ANNiris = nn.Sequential(
        nn.Linear(4,nHidden),               # input layer 
        nn.ReLU(),                     # activation
        nn.Linear(nHidden,nHidden),              # hidden layer
        nn.ReLU(),                     # activation
        nn.Linear(nHidden,3),               # output layer     
        )

    # loss function
    lossfun = nn.CrossEntropyLoss()    # this combines the Softmax function and the loss function

    # optimizer
    optimizer = torch.optim.SGD(ANNiris.parameters(), lr= learningRate)
    
    # output
    return ANNiris,lossfun,optimizer
    

#%% Function for training

def trainTheModel(ANNiris):
    
    # training 
    for epochi in range(numepochs):
        
        # forward pass
        yHat = ANNiris(data)
        
        # compute loss
        loss = lossfun(yHat, labels)
        
        #backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # training loop complete 

    # final forward pass
    predictions = ANNiris(data)

    predlabels = torch.argmax(predictions, axis=1)
    return 100*torch.mean((predlabels == labels).float())


#%% Running the model

# initalise variables
numepochs = 150
numhidden = np.arange(1,129)
accuracies = []

for nunits in numhidden:
    
    # create a fresh model instance 
    ANNiris, lossfun, optimizer = createIrisModel(nunits)
    
    # run the model
    acc = trainTheModel(ANNiris)
    accuracies.append(acc)


#%% Visualise accuracy

# report accuracy
fig,ax = plt.subplots(1,figsize=(12,6))

ax.plot(accuracies,'ko-',markerfacecolor='w',markersize=9)
ax.plot(numhidden[[0,-1]],[33,33],'--',color=[.8,.8,.8])
ax.plot(numhidden[[0,-1]],[67,67],'--',color=[.8,.8,.8])
ax.set_ylabel('accuracy')
ax.set_xlabel('Number of hidden units')
ax.set_title('Accuracy')
plt.show()






















