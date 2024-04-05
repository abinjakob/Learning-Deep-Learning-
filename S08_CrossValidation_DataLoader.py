#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 20:08:36 2023

Basic Python Coding from Deep Learning Tutorial by Mike Cohen 
Section 08 - Cross Validation : DataLoader

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

#%% creating the training data and test data, then creating batches

# using scikitlearn for splitting into train and test data
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, train_size= .8)  # using the train size

# convert into PyTorch Datasets (overwriting the variables) 
# Note: need to do this intermediatory step before using dataloader so that the labels are not detached
train_data = TensorDataset(train_data, train_labels)
test_data  = TensorDataset(test_data, test_labels)

# translate to DataLoader objects 
train_loader = DataLoader(train_data, shuffle= True, batch_size= 12)                # batched with 12 in each -> so 10 batches
test_loader  = DataLoader(test_data, batch_size= test_data.tensors[0].shape[0])     # all of the data is batched into 1 -> so 1 batch

#%% function to create ANN model

def createANewModel():
    
    # model architecture
    ANNiris = nn.Sequential(
        nn.Linear(4, 64),       # input layer 
        nn.ReLU(),              # activation unit
        nn.Linear(64, 64),      # hidden layer 
        nn.ReLU(),              # activation layer 
        nn.Linear(64, 3),       # output layer 
        )
    
    # loss function 
    lossfun = nn.CrossEntropyLoss()
    
    # optimizer 
    optimizer = torch.optim.SGD(ANNiris.parameters(), lr= 0.1)
    
    return ANNiris, lossfun, optimizer

#%% train the model

numepochs = 500


# function to train the model
def trainTheModel():
    
    # initialise accuracies 
    trainAcc = []
    testAcc  = []
    
    # loop over epochs
    for epochi in range(numepochs):
        
        # to store the accuracy of each individual mini-batch
        batchAcc =[]
        
        # loop over training batches 
        for X,y in train_loader:
            
            # forward pass
            yHat = ANNiris(X)
            
            # compute loss
            loss = lossfun(yHat, y)
            
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # compute training accuracy just for the current batch
            batchAcc.append( 100 * torch.mean((torch.argmax(yHat, axis= 1) == y).float()).item() )
        
        # end of batch loop
        
        # getting average training accuracy 
        trainAcc.append(np.mean(batchAcc))
        
        # test accuracy
        X,y = next(iter(test_loader))       # extract X,y from dataloader using nest-iter instead of for-loop
        predlabels = torch.argmax(ANNiris(X), axis= 1)
        testAcc.append( 100 * torch.mean((predlabels == y).float()).item() )
        
    # function output
    return trainAcc, testAcc

#%% Modeling 

# create a model
ANNiris, lossfun, optimizer = createANewModel()

# train the model
trainAcc, testAcc = trainTheModel()

#%% plot the results
fig = plt.figure(figsize=(10,5))

plt.plot(trainAcc,'ro-')
plt.plot(testAcc,'bs-')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend(['Train','Test'])

# optional zoom-in to final epochs
# plt.xlim([300,500])
# plt.ylim([90,100.5])

plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
        
            
        
        


































