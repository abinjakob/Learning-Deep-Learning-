#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 13:15:39 2023

Basic Python Coding from Deep Learning Tutorial by Mike Cohen 
Section 08 - Cross Validation : Scikitlearn

@author: abinjacob
"""

#%% Libraries 

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# for scikitlearn
from sklearn.model_selection import train_test_split

# for iris dataset
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

#%% function that creates ANN model

def createANewModel():
    
    # model architecture 
    ANNiris = nn.Sequential(
        nn.Linear(4, 64),           # input layer
        nn.ReLU(),                  # activation unit
        nn.Linear(64, 64),          # hidden layer
        nn.ReLU(),                  # activation unit
        nn.Linear(64, 3),           # output layer     
        )
    
    # loss function
    lossfun = nn.CrossEntropyLoss()
    
    # optimizer
    optimizer = torch.optim.SGD(ANNiris.parameters(), lr= .01)
    
    
    return ANNiris, lossfun, optimizer

#%% train the model

# global variable
numepochs = 200

def trainTheModel(trainProp):
    
    # initialise losses
    losses   = torch.zeros(numepochs)
    trainAcc = []
    testAcc  = []
    
    # seperate train from the test data using scikitlearn function -> treain_test_split
    X_train, x_test, y_train, y_test = train_test_split(data, labels, train_size = trainProp)
    
    # loop over epochs
    for epochi in range(numepochs):
        
        # seperate train from the test data using scikitlearn function -> treain_test_split
        # Note 1: unique split for each prop
        # this is basically OVERFITTING because the items in test set could be in the training set in the next loop (epochs) !!
        # -> the ideal way is to place this before the for loop
        # Note 2: using training size and not test size 
        # X_train, x_test, y_train, y_test = train_test_split(data, labels, train_size = trainProp)
            
            
        # forward pass and loss
        yHat = ANNiris(X_train)
        loss = lossfun(yHat, y_train)
        
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # compute training accuracy 
        trainAcc.append( 100*torch.mean((torch.argmax(yHat, axis= 1) == y_train).float()).item() )
        
        #test accuracy
        predlabels = torch.argmax( ANNiris(x_test), axis= 1 )
        testAcc.append( 100*torch.mean((predlabels == y_test).float()).item() )
        
        # -> we are using the test set inside the training loop just to look the accuracy and nowhere it is used 
        # in the backprop to train the model and hence this is completely fine (we are not overfitting!!)
        
    # function output
    return trainAcc, testAcc

#%% testing the model by running once 

# create a model
ANNiris, lossfun, optimizer = createANewModel()

# train the model
trainAcc, testAcc = trainTheModel(.8)                           # here the input is training proportion

#%% plot the results
fig = plt.figure(figsize=(10,5))

plt.plot(trainAcc,'ro-')
plt.plot(testAcc,'bs-')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend(['Train','Test'])
plt.show()
        
#%% running the experiment 

trainSetSizes = np.linspace(.2,.95,10)                          # testing training proportion from 2% to 95% 

allTrainAcc = np.zeros((len(trainSetSizes),numepochs))
allTestAcc = np.zeros((len(trainSetSizes),numepochs))

for i in range(len(trainSetSizes)):
  
  # create a model
  ANNiris,lossfun,optimizer = createANewModel()
  
  # train the model
  trainAcc,testAcc = trainTheModel(trainSetSizes[i])
  
  # store the results
  allTrainAcc[i,:] = trainAcc
  allTestAcc[i,:] = testAcc

        
#%% plotting the results

fig,ax = plt.subplots(1,2,figsize=(13,5))

ax[0].imshow(allTrainAcc,aspect='auto',
             vmin=50,vmax=90, extent=[0,numepochs,trainSetSizes[-1],trainSetSizes[0]])
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Training size proportion')
ax[0].set_title('Training accuracy')

p = ax[1].imshow(allTestAcc,aspect='auto',
             vmin=50,vmax=90, extent=[0,numepochs,trainSetSizes[-1],trainSetSizes[0]])
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Training size proportion')
ax[1].set_title('Test accuracy')
fig.colorbar(p,ax=ax[1])

plt.show() 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


































