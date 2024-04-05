#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 17:51:54 2023

Basic Python Coding from Deep Learning Tutorial by Mike Cohen 
Section 05 - Math,Numpy,PyTorch
Contains:
    - Transpose
    - Dot Product
    - Matrix Multiplication
    - Softmax
    - Logarithm
    - Entropy and Cross-entropy
    - Min, Max, Argmin, Argmax
    - Mean and Variance 
    - Random sampling and sampling variability
    - Reproduce randomness using seeding
    - T-test using scipy
    - Derivatives
    - Derivatives: Product and Chain rule

@author: abinjacob
"""

#%% Libraries

import numpy as np
import torch 
import matplotlib.pyplot as plt

# importing torch library for neural networks 
import torch.nn as nn

# for different functions
import torch.nn.functional as F

# for t-test etc.
import scipy.stats as stats 

# sympy = symbolic math in python (Not much used in deep python)
import sympy as sym
import sympy.plotting.plot as symplot


#%% Transpose using Numpy

# For a Vector
nv = np.array([ [1,2,3,4] ])
print(nv)

# transposing the nd array
print(nv.T)



# For a Matrix
nm = np.array([ [1,2,3,4],[5,6,7,8] ])
print(nm)
print(nm.T)


#%% Transpose usin PyTorch 

# For a Vector 
tv = torch.tensor([ [1,2,3,4] ])

# transposing tensor 
print(tv.T)



# For a Matrix 
tm = torch.tensor([ [1,2,3,4],[5,6,7,8] ])

# transposing tensor 
print(tm.T)


#%% Dot Product

# Using Numpy
# creatign a numpy vector
nv1 = np.array([1,2,3,4])
nv2 = np.array([0,1,0,-1])

print(np.dot(nv1,nv2))



# Using PyTorch
tv1 = torch.tensor([1,2,3,4])
tv2 = torch.tensor([0,1,0,-1])

print(torch.dot(tv1,tv2))


#%% Matrix Multiplication

A = np.random.randn(3,4)
B = np.random.randn(4,5)
C = np.random.randn(3,7)

# numpy matrix multiplication 
ans1 = A@B

# rounding off
ans2 = np.round(A@B,2)

# A*C
ans3 = np.round(A@C,2)                  
# not possible as inner dimensions are different 
# So transposing A to match the inner dimensions

ans3 = np.round(A.T@C,2)                  



# PyTorch matrix multiplication 
A = torch.randn(3,4)                            # creating a random matrix 
B = torch.randn(4,5)

C = np.random.randn(4,7)                        # creating a random numpy matric to see if pytorch matrix * numpy matrix is possible 
Ct = torch.tensor(C, dtype = torch.float)       # converting C into a PyTorch Tensor

ans  = np.round(A@B,2)
ans2 = np.round(A@C,2)                          # pytorch matrix * numpy matrix works!!!
ans3 = np.round(A@Ct,2)


#%% Softmax

### calculating softmax in numpy 
# manual method | formula: sigma(i) = e^z(i)/sum(e^z)
z = [1,2,3]

# computing softmax 
num = np.exp(z)
den = np.sum ( np.exp(z) )

sigma = num/den

print(np.round(sigma,3))
print(np.sum(sigma))


# same usng a larger integers
z = np.random.randint(-5,high=15,size=25)
# run the above functions
# plotting 
plt.plot(z,sigma,'ko')




### softmax using PyTorch 
# here it uses torch.nn

# creating an instance of the softmax activation class
softfun = nn.Softmax(dim=0)

# applying data to the function
sigmaT = softfun(torch.Tensor(z))           # also has  to covert the z which is a list to tensor

print(sigmaT)
print(torch.sum(sigmaT))



#%% Logarithm

# creating a set of points to evaluate
x= np.linspace(.0001,1,200)

# computing log
logx= np.log(x)

# plotting 
fig = plt.figure()
plt.plot(x,logx,'ks-',markerfacecolor='w')

#%% Entropy and Cross-entropy using Numpy


# always it should have all the values of probability to calculate entropy, That is:

          # probability of event not happening is 1 - 0.25  = 0.75
x = [.25,.75]
     # probability of event happening 

H = 0
for p in x:
    H += -p * np.log(p)

# can also write as below when there are eactly 2 events 
p = 0.25
H = -(p*np.log(p) + (1-p)*np.log(1-p))
# the above is the binary cross entropy 


### for cross-entropy

# all probabilty sum must be 1 
p = [1,0]       # sum = 1
q = [.25,.75]   # sum = 1 


H = 0
for i in range(len(p)):
    H += - (p[i]*np.log(q[i]))

print(f' Cross entropy is {H}')

#%% Entropy and Cross-entropy using PyTorch
# requires to import torch.nn.functional as F

# inputs must be Tensors 
p_tensor = torch.Tensor(p)
q_tensor = torch.Tensor(q)

F.binary_cross_entropy(q_tensor, p_tensor)


#%% Min, Max, Argmin, Argmax

# create a vector 
v = np.array([1,40,2,-3])

# maximum and minimum values 
minval = np.min(v)                                                  # gets the lowest value in the vector
maxval = np.max(v)                                                  # gets the higest value in the vector

print(f'Min is {minval} and Max val is {maxval}')

# index of the min and max 
minidx = np.argmin(v)                                              # gets the location of the lowest value in the vector
maxidx = np.argmax(v)                                              # gets the location of the highest value in the vector

print(f'Min Location is {minidx} and the Max Location is {maxidx}')


# for a matrix 
M = np.array([ [0,1,10], 
               [20,8,5] ]) 

# finding min values in matrix 
minval1 = np.min(M)                               # calculate the min value for the entire matrix 
minval2 = np.min(M,axis=0)                        # calculate the min vlaue in each column (across rows)
minval3 = np.min(M,axis=1)                        # calculate the min value in each rows (across column)


print(f'Min value in the entire matrix is {minval1}')
print(f'Min value in each column is {minval2}')
print(f'Min value in each row is {minval3}')  


# finding location of min values in matrix 
minidx1 = np.argmin(M)                            # location of min value for the entire matrix 
minidx2 = np.argmin(M,axis=0)                     # location of min vlaue in each column (across rows)
minidx3 = np.argmin(M,axis=1)                     # location of min value in each rows (across column)

print(minidx1)
print(minidx2)
print(minidx3)  



### Using PyTorch
# creating a vector 
v = torch.tensor([1,40,2,-3])

# find min and max
minval = torch.min(v)
maxval = torch.max(v)

# find locations of min and max values 
minidx = torch.argmin(v)
maxidx = torch.argmax(v)


# for Matrix 
M = torch.tensor([ [0,1,10],
                   [20,8,5]])

min1 = torch.min(M)
min2 = torch.min(M, axis=0)                     # for each column. But returns vlaues and its indexes 
print(min2.values)                              # gives the min values in each columns 
print(min2.indices)                             # gives the location of the min values in each columns


#%% Mean and Variance 

### Numpy
x = [1,2,4,6,5,4,0]
n = len(x)

# calculating mean 
mean1 = np.mean(x)                              # mean function 
mean2= np.sum(x)/n                              # manual method 

# variance 
var1 = np.var(x, ddof=1)                        
# -- has to set the degrees of freedom to 1, that is 1/(n-1) in formula (unbiased variance)
# -- by default the degrees of freedom is 0 in python, ie. 1/n in formula which is a biased variance 


#%% Random sampling and sampling variability

# a list with random numbers 
x = [1,2,4,6,5,4,0,-4,5,-2,6,10,-9,1,3,-6]
n = len(x)

# computing population mean
popmean = np.mean(x)

# compute a sample mean
sample = np.random.choice(x, size=5,replace=True)                 # choosing random 5 numbers from x
sampmean = np.mean(sample)

# you can see that they are not equal
print(popmean,sampmean)



#### compute lots of sample mean for above data 

# number of experiments to run
nExpers =10000

# run experiment 
sampleMeans = np.zeros(nExpers)

for i in range(nExpers):
    
    # draw a sample
    sample = np.random.choice(x,size=5,replace=True)            # larger the size better it is
    
    # compute its mean 
    sampleMeans[i] = np.mean(sample)
    

# plotting in a histogram
plt.hist(sampleMeans,bins=40,density= True)
plt.plot([popmean,popmean],[0,.3],'m--')


#%% Reproduce randomness using seeding

# will produce random number always
print(np.random.randn(5))

# fixing the seed (Old Method)
np.random.seed(17)                  # 17 is like a arbitory number we put so that every time we use 17 we get the same randomness
print(np.random.randn(5))
print(np.random.randn(5))

#%% T-test using scipy

# creating a random data 
# parameters 
n1  = 30            # samples in dataset 1
n2  = 40            # samples in dataset 2
mu1 = 1             # popluation mean in dataset 1
mu2 = 2             # popluation mean in dataset 2

data1 = mu1 + np.random.randn(n1)
data2 = mu2 + np.random.randn(n2)

# plotting them
plt.plot(np.zeros(n1),data1,'ro',markerfacecolor='w',markersize=14)
plt.plot(np.ones(n2),data2,'bs',markerfacecolor='w',markersize=14)
plt.xlim([-1,2])


# t-test via stats package (here _ind means independent samples)
t,p = stats.ttest_ind(data1,data2)

print(t,p)


#%% Derivatives

# Need to import the following (not much used for DL. Just for the representation purposes)
# import sympy as sym
# import sympy.plotting.plot as symplot


# creating symbolic variable 
x = sym.symbols('x')

# create a function 
fx = 2*x**2

# compute the derivative 
df = sym.diff(fx,x)

print(fx)
print(df)

# plotting using symplot
symplot(fx,(x,-4,4), title = 'Function')
symplot(df,(x,-4,4), title = 'Derivative')

#%% Derivatives: Product and Chain rule 

# creating symbolic variable 
x = sym.symbols('x')

# create two function 
fx = 2*x**2
gx = (4*x**3) - (3*x**4)

# compute the derivatives 
df = sym.diff(fx)
df = sym.diff(gx)





































