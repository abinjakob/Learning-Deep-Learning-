#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 17:03:44 2023

Basic Python Coding from Deep Learning Tutorial by Mike Cohen 
Section 06 - Gradient Descent
Contains:
    - Gradient Descent in 1D
    
@author: abinjacob
"""

#%% Libraries 
import numpy as np
import matplotlib.pyplot as plt

from IPython import display
display.set_matplotlib_formats('svg')


#%% Gradient Descent in 1D
# everything is done manually using numpy and not using inbuilt functios of PyTorch at the moment 


# function (as a function)
def fx(x):
    return 3*x**2 - 3*x + 4

# derivative function
def deriv(x):
    return 6*x - 3

# defining x 
x = np.linspace(-2,2,2001)

# plotting  
plt.plot(x,fx(x),x,deriv(x))
plt.xlim(x[[0,-1]])
plt.grid()                                      # turning on grids in the plot
plt.xlabel('x')
plt.ylabel('f(x)')


###---- learning algorithm

# setting a random starting point
localmin = np.random.choice(x,1)                # pick a random choice from x 
print(f'first estimate {localmin}')             # just printing the first local min choosen 

# learning parameters 
learning_rate = .01
training_epochs = 100

# loop to run through the training --> Gradient Descent Algorithm 
for i in range(training_epochs):
    grad = deriv(localmin)
    localmin = localmin - (learning_rate * grad)
    
localmin 


# plot the results
plt.plot(x,fx(x),x,deriv(x))
plt.plot(localmin,deriv(localmin),'ro')
plt.plot(localmin,fx(localmin),'ro')

plt.xlim(x[[0,-1]])
plt.grid()                                      # turning on grids in the plot
plt.xlabel('x')
plt.ylabel('f(x)')
    
       
#----- just to understand how derivatives work running everything again

# setting a random starting point
localmin = np.random.choice(x,1)                # pick a random choice from x 

# learning parameters 
learning_rate = .01
training_epochs = 100

modelparams = np.zeros((training_epochs,2))

# loop to run through the training --> Gradient Descent Algorithm 
for i in range(training_epochs):
    grad = deriv(localmin)
    localmin = localmin - (learning_rate * grad)
    
    # storing each values 
    modelparams[i,:] = localmin,grad


# plot the gradient over iterations 

fig,ax = plt.subplots(1,2,figsize=(12,4))

for i in range(2):
    ax[i].plot(modelparams[:,i],'o-')
    ax[i].set_xlabel('iteration')

ax[0].set_ylabel('Local Minimum')
ax[1].set_ylabel('Derivative')
    
# now play around with learning_rate = .01
    

#%% Code Challenge 

# function (as a function)
def fx(x):
    return np.cos(2* np.pi * x) + x**2

# derivative function
def deriv(x):
    return (2 * x) - (2 * np.pi * np.sin (2 * np.pi * x))

# defining x 
x = np.linspace(-2,2,2001)



###-- learning algorithm

# setting a random local minima to start with  
localmin = np.random.choice(x,1)
print(localmin)

# setting parameters 
learning_rate = 0.1
training_epochs = 100

# just for understanding the learning 
modelparams = np.zeros((training_epochs,2))

for i in range(training_epochs):
    grad = deriv(localmin)
    localmin = localmin - (grad * learning_rate)
    # storing each values to understand the learning
    modelparams[i,:] = localmin,grad
    
localmin


# now hardcoding starting point for local minima as 0
localmin = np.array([0])


# setting parameters 
learning_rate = 0.1
training_epochs = 100

# just for understanding the learning 
modelparams = np.zeros((training_epochs,2))

for i in range(training_epochs):
    grad = deriv(localmin)
    localmin = localmin - (grad * learning_rate)
    # storing each values to understand the learning
    modelparams[i,:] = localmin,grad
    
localmin








    
    
    
    
    
