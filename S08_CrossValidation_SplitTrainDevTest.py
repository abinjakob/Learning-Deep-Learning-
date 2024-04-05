#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 19:39:38 2023

Basic Python Coding from Deep Learning Tutorial by Mike Cohen 
Section 08 - Cross Validation : Splitting data into train, devset and test

Will be splitting the data manually and also using scikitlearn 
A dummy data will be used for the demonstartion purpose

@author: abinjacob
"""

#%% libraries 

import numpy as np
from sklearn.model_selection import train_test_split

#%% creating a dummy dataset

fakedata = np.tile(np.array([1,2,3,4]), (10,1)) + np.tile(10 * np.arange(1,11),(4,1)).T
fakelabels = np.arange(10)>4
print(fakedata)
print('')
print(fakelabels)

#%% using scikitlearn train_test_split

# specify size of partition (train, devset, test)
partitions = [.8,.1,.1]

# split the data 
# note: the train_test_split does not allow to natively split the data into three
#       so need to run the train_test_split twice:
#       First: split the data into train and test set (containing 20%)
#       Second: split the test set further into devset and test set

# first split
# testTMP_data and testTMP_labels are temporary data which will be further divided in second split to test and dev sets
train_data, testTMP_data, train_labels, testTMP_labels = train_test_split(fakedata, fakelabels, train_size= partitions[0])

# second split 
# need to divide the testTMP_data into 50-50 (which is 10% and 10% of actual data) 
split = partitions[1]/ np.sum(partitions[1:])
devset_data, test_data, devset_labels, test_labels = train_test_split(testTMP_data, testTMP_labels, train_size= split)


# print the sizes 
print(f'Training data size: {train_data.shape}')
print(f'Devset data size: {devset_data.shape}')
print(f'Test data size: {test_data.shape}')

print('Train data')
print(train_data)
print('')

print('Dev data')
print(devset_data)
print('')

print('Test data')
print(test_data)
print('')

#%% splitting data manually using numpy

# specify size of partition (train, devset, test)
partitions = np.array([.8,.1,.1])

# note: there are many ways to achieve this random picking based on the desired proportions
# here the data indices are randomised and then picked first 8 indices data into data set and 9th indice into
# dev set and 10th indice into test set

# convert the partitions into integer boundaries 
# np.cumsum return the cumulative sum of the elements along a given axis.
# .astype(int) converts into datatype int
partitionBnd = np.cumsum(partitions * len(fakelabels)).astype(int)

# creating random indices 
randIndices = np.random.permutation(range(len(fakelabels)))

# select training set
train_dataN = fakedata[randIndices[:partitionBnd[0]],:]
train_labelsN = fakelabels[randIndices[:partitionBnd[0]]]

# select dev set
devset_dataN = fakedata[randIndices[partitionBnd[0]:partitionBnd[1]],:]
devset_labelsN = fakelabels[randIndices[partitionBnd[0]:partitionBnd[1]]]

# select test set
test_dataN = fakedata[randIndices[partitionBnd[1]:partitionBnd[2]],:]
test_labelsN = fakedata[randIndices[partitionBnd[1]:partitionBnd[2]]]

# print the sizes 
print(f'Training data size: {train_dataN.shape}')
print(f'Devset data size: {devset_dataN.shape}')
print(f'Test data size: {test_dataN.shape}')

print('Train data')
print(train_dataN)
print('')

print('Dev data')
print(devset_dataN)
print('')

print('Test data')
print(test_dataN)
print('')

