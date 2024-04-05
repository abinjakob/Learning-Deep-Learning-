#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 16:19:01 2023

Basic Python Coding from Deep Learning Tutorial by Mike Cohen 
Section 06 - Gradient Descent - Part2
Contains:
    - Gradient Descent in 2D
    -  Gradient Ascent


@author: abinjacob
"""

#%% Libraries 

import numpy as np
import matplotlib.pyplot as plt

# sympy to compute partial derivatives 
import sympy as sym

from IPython import display
display.set_matplotlib_formats('svg')



#%% Gradient Descent in 2D

# the 'peaks' function
def peaks(x,y):
    # expand to a 2D mesh 
    x,y = np.meshgrid(x,y)
    
    z = 3*(1-x)**2 * np.exp(-(x**2) - (y+1)**2) \
    - 10*(x/5 - x**3 - y**5) * np.exp(-x**2-y**2) \
    - 1/3*np.exp(-(x+1)**2 - y**2)
    
    return z

# create the  landscape
x = np.linspace(-3,3,201)
y = np.linspace(-3,3,201)

Z = peaks(x,y)

# plotting the function to have a look at it
plt.imshow(Z,extent=[x[0],x[-1],y[0],y[-1]], vmin=-5, vmax=5, origin='lower')


# create derivative using sympy
# need to create the function again as it has to be in sympy fromat

sx,sy = sym.symbols('sx,sy')

sZ = 3*(1-sx)**2 * sym.exp(-(sx**2) - (sy+1)**2) \
      - 10*(sx/5 - sx**3 - sy**5) * sym.exp(-sx**2-sy**2) \
      - 1/3*sym.exp(-(sx+1)**2 - sy**2)
      

# create functions from the sympy-computed derivatives 
# it looks a bit complicated because the function above is bit complicated
df_x = sym.lambdify( (sx,sy),sym.diff(sZ,sx),'sympy' )                          # sym.lambsify actual converts the sym to numpy so that we can call later         
df_y = sym.lambdify( (sx,sy),sym.diff(sZ,sy),'sympy' )

df_x(1,1).evalf()               # just calculating the partial derivative of x at 1,1


##--- TRAINING

# random starting point (uniform between -2 and +2)
localmin = np.random.rand(2)*4-2 # also try specifying coordinates
startpnt = localmin[:] # make a copy, not re-assign

# learning parameters
learning_rate = .01
training_epochs = 1000

# run through training
trajectory = np.zeros((training_epochs,2))
for i in range(training_epochs):
  grad = np.array([ df_x(localmin[0],localmin[1]).evalf(), 
                    df_y(localmin[0],localmin[1]).evalf() 
                  ])
  localmin = localmin - learning_rate*grad  # add _ or [:] to change a variable in-place
  trajectory[i,:] = localmin


print(localmin)
print(startpnt)

# let's have a look!
plt.imshow(Z,extent=[x[0],x[-1],y[0],y[-1]],vmin=-5,vmax=5,origin='lower')
plt.plot(startpnt[0],startpnt[1],'bs')
plt.plot(localmin[0],localmin[1],'ro')
plt.plot(trajectory[:,0],trajectory[:,1],'r')
plt.legend(['rnd start','local min'])
plt.colorbar()
plt.show()



#%% CODE CHALLENGE - Gradient Ascent 


##--- TRAINING

# random starting point (uniform between -2 and +2)
localmax = np.random.rand(2)*4-2 # also try specifying coordinates

# learning parameters
learning_rate = .01
training_epochs = 1000

# run through training
for i in range(training_epochs):
  grad = np.array([ df_x(localmax[0],localmax[1]).evalf(), 
                    df_y(localmax[0],localmax[1]).evalf() 
                  ])
  localmax = localmax + learning_rate*grad 









      