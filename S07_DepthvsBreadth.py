#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 18:30:41 2023

Basic Python Coding from Deep Learning Tutorial by Mike Cohen 
Section 07 - Artificial Neural Networks : Depth vs Breadth

@author: abinjacob
"""

#%% Libraries 

import torch
import torch.nn as nn
import numpy as np

from torchsummary import summary

#%% build two models

# model 01 
widenet = nn.Sequential(
    nn.Linear(2,4),                  # hidden layer
    nn.Linear(4,3),                  # output layer
    )

# model 02
deepnet = nn.Sequential(
    nn.Linear(2,2),                  # hidden layer
    nn.Linear(2,2),                  # hidden layer
    nn.Linear(2,3),                  # output layer
    )

#%% check out the parameters 

for p in deepnet.named_parameters():
    print(p)
    print(' ')
    
#%% count the number of nodes
# since every node will have a bias term we can count the ndoes by counting the bias term

numNodesInWide = 0
for p in widenet.named_parameters():
    
    if 'bias' in p[0]:
        numNodesInWide += len(p[1])
        
numNodesInDeep = 0

# just having menaningful variable names below just for understanding
for paramName, paramVect in deepnet.named_parameters():
    if 'bias' in paramName:
        numNodesInDeep += len(paramVect)
        
print(f'There are {numNodesInWide} in Wide Network')
print(f'There are {numNodesInDeep} in Deep Network')


#%% counting the total number of trainable parameters (ie. the weights)

#  for model 1
nparams = 0

for p in widenet.parameters():
    if p.requires_grad:
        print(f'This piece has {p.numel()} parameters')
        nparams += p.numel()
        
print(f'Total of parameters is {nparams}')



#  for model 2
nparams = 0

for p in deepnet.parameters():
    if p.requires_grad:
        print(f'This piece has {p.numel()} parameters')
        nparams += p.numel()
        
print(f'Total of parameters is {nparams}')


#%% simple way to print out the model info

# need to import summary from torchsummary 
summary(widenet,(1,2))
