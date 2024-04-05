#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 16:01:50 2023

Basic Python Coding from Deep Learning Tutorial by Mike Cohen 
Section 06 - Gradient Descent - Part3
Contains:
    - Paramertic Experiments on GD


@author: abinjacob
"""

#%% Libraries 

import numpy as np
import matplotlib.pyplot as plt 

from IPython import display
display.set_matplotlib_formats('svg')


#%% Paramertic Experiments on GD

# function
x  = np.linspace(-2*np.pi,2*np.pi,401)
# fx = np.sin(x) *  np.exp(-x** 2 * .05)

# # derivative of the funciotn 
# df = np.cos(x) * np.exp(-x** 2 * .05) + np.sin(x)*(-.1*x)* np.exp(-x** 2 * .05)

# # plotting for inspection
# plt.plot(x,fx,x,df)
# plt.legend(['f(x)','df'])


# the above line was just to show how the fucntion looks like 
# now converting this into a python function 

# for the function
def fx(x):
    return np.sin(x) *  np.exp(-x** 2 * .05)

# for the derivative function
def deriv(x):
    return np.cos(x) * np.exp(-x** 2 * .05) + np.sin(x)*(-.1*x)* np.exp(-x** 2 * .05)


# setting random starting point for the local minima 
localmin = np.random.choice(x,1)

# setting parameters 
learning_rate = .01
training_epochs = 1000

# running training 
for i in range(training_epochs):
    grad = deriv(localmin)
    localmin = localmin - (grad*learning_rate)

# plotting the results 
plt.figure()
plt.plot(x,fx(x),x,deriv(x),'--')
plt.plot(localmin,deriv(localmin),'ro')
plt.plot(localmin,fx(localmin),'ro')
plt.grid()

#%% Experiment 1: Systematically vary the starting locations

startlocs = np.linspace(-5,5,50)             # trying various starting locations
finalres = np.zeros(len(startlocs))

# loop over starting points 
for idx, localmin in enumerate(startlocs):
    
    # running training
    for i in range (training_epochs):
        grad = deriv(localmin)
        localmin = localmin - (learning_rate * grad)
    
    # store the final guess 
    finalres[idx] = localmin 

# plotting the sratlocs and finalres 
plt.plot(startlocs,finalres,'s-')

#%% Experiment 2: systematically varying the learning rates 

learningrates = np.linspace(1e-10,1e-1,50)      # trying various learning rates
finalres = np.zeros(len(learningrates))

# loop over the learning rates 
for idx, learningRates in enumerate(learningrates):
    
    # force stratign the localmin guess to 0
    # so that we can see determining the effect of learning rate by controlling the localmin 
    localmin = 0
    
    # running training 
    for i in range(training_epochs):
        grad = deriv(localmin)
        localmin = localmin - (grad * learningRates)
    
    finalres[idx] = localmin
    
plt.plot(learningrates,finalres,'s-')

#%% Experiment 3: interaction between learning rate and training epochs 

learningrates = np.linspace(1e-10,1e-1,50)                  # trying various learning rates
training_epochs = np.round(np.linspace(10,500,40))          # also trying different training epochs

# creating a matrix to store the results 
finalres = np.zeros((len(learningrates), len(training_epochs)))


# looping over learnig rates
for Lidx,learningRate in enumerate(learningrates):
    
    # looping over training epochs 
    for Eidx, trainingEpochs in enumerate(training_epochs):
        
        # setting start point to 0
        localmin = 0
        
        # running training 
        for i in range(int(trainingEpochs)):
            grad = deriv(localmin)
            localmin = localmin - (grad * learningRate)
        
        # store the result
        finalres[Lidx,Eidx] = localmin
        
#%% CODE CHALLENGE: Fixed vs Dynamic Learning Rate 

def fx(x):
    return 3*x**2 - 3*x + 4

# derivative function
def deriv(x):
    return 6*x - 3

# defining x 
x = np.linspace(-2,2,2001)


# setting a random starting point
localmin = np.random.choice(x,1)                # pick a random choice from x 

# parameters 
training_epochs = 100

# running training 
for i in range(training_epochs):
    
    # dynamic changing rate
    if training_epochs < 50:
        learning_rate = .01
        grad = deriv(localmin)
        localmin = localmin - (grad*learning_rate)
    
    else:
        learning_rate = .1
        grad = deriv(localmin)
        localmin = localmin - (grad*learning_rate)
        

        
    



































