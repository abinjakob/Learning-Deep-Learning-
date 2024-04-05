#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 20:24:47 2023

Basic Python Coding from Deep Learning Tutorial by Mike Cohen 
Section 09 - Regularization : Dropout Regularization


@author: abinjacob
"""

#%% libraries

import torch 
import torch.nn as nn
import torch.nn.functional as F

#%% using dropout 

# define a dropout instance and make some data
prob = .5                   
# note: by default the probability is .5 and need not mention it explicitly
dropout = nn.Dropout(p = prob)

# dummy data
x = torch.ones(10)

# seeing what droput returns 
y = dropout(x)
print(x)
print(y)
print(torch.mean(y))

#%% eval() mode
# note: dropout is turned off when in evaluation mode

dropout.eval()
y = dropout(x)

# seeing what droput returns during eval
print(x)
print(y)
print(torch.mean(y))



