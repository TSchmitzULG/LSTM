#!/usr/bin/python
# -*- coding: latin-1 -*-
"""datashaping loading in file at each batch, format [batchSize,numStep]"""
import scipy.io as sio
import numpy as np


# take input matrix and return shuffled train/test input/input
def splitShuffleData(matrix,num_step,trainTestRatio,maxSize):
    matrixIn = matrix[:maxSize,0]
    matrixOut = matrix[:maxSize,1]
    trainSize = int(len(matrixIn)*trainTestRatio)
    reshapedInput  = []
    reshapedOutput = []
    nbExample = len(matrixIn)-num_step
    for i in range(nbExample):
        temp_list = matrixIn[i:i+num_step]
        temp_list_out = [matrixOut[i+num_step-1]]
        reshapedInput.append(np.array(temp_list))
        reshapedOutput.append(np.array(temp_list_out))
    inputShuffled = shuffleMatrix(reshapedInput)
    outputShuffled = shuffleMatrix(reshapedOutput)
    trainInput = inputShuffled[:trainSize]
    trainOutput = outputShuffled[:trainSize]
    testInput = inputShuffled[trainSize:]
    testOutput = outputShuffled[trainSize:]
    return trainInput,trainOutput,testInput,testOutput

def shapeData(matrix,num_step,maxSize):
    matrixIn = matrix[:maxSize,0]
    matrixOut = matrix[:maxSize,1]
    reshapedInput  = []
    reshapedOutput =[]
    nbExample = len(matrixIn)-num_step
    for i in range(nbExample):
        temp_list = matrixIn[i:i+num_step]
        temp_list_out = [matrixOut[i+num_step-1]]
        reshapedInput.append(np.array(temp_list))
        reshapedOutput.append(np.array(temp_list_out))
    return reshapedInput, reshapedOutput 

def shuffleMatrix(data):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    return [data[i] for i in shuffled_indices]



