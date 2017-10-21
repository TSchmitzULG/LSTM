#!/usr/bin/python
# -*- coding: latin-1 -*-
"""datashaping5  loading in file at each batch and give separate features for gain parameters, format [batchSize,numStep]"""
import scipy.io as sio
import numpy as np


# take input matrix and return shuffled train/test input/input
def splitShuffleData(matrix,num_step,trainTestRatio,maxSize):
    matrixIn = matrix[:maxSize,0]
    matrixParameters = matrix[:maxSize,1]
    #matrixParameters = [matrixParameters[i]*0.1 for i in range(len(matrixParameters))]
    matrixOut = matrix[:maxSize,2]
    trainSize = int(len(matrixIn)*trainTestRatio)
    reshapedInput  = []
    reshapedOutput =[]
    reshapedGain = []
    nbExample = len(matrixIn)-num_step
    for i in range(nbExample):
        temp_list = []
        temp_list_out = []
        for j in range(num_step):
            temp_list.append(matrixIn[i+j])
        vectorOut = []
        vectorGain = []
        vectorOut.append(matrixOut[i+num_step-1])
        vectorGain.append(matrixParameters[i+num_step-1])
        reshapedInput.append(np.array(temp_list))
        reshapedOutput.append(np.array(vectorOut))
        reshapedGain.append(np.array(vectorGain))
    inputShuffled = shuffleMatrix(reshapedInput)
    outputShuffled = shuffleMatrix(reshapedOutput)
    gainShuffled = shuffleMatrix(reshapedGain)
    trainInput = inputShuffled[:trainSize]
    trainOutput = outputShuffled[:trainSize]
    trainGain = gainShuffled[:trainSize]
    testGain = gainShuffled[trainSize:]
    testInput = inputShuffled[trainSize:]
    testOutput = outputShuffled[trainSize:]
    return trainInput,trainOutput,trainGain,testInput,testOutput,testGain

def shapeData(matrix,num_step,maxSize):
    matrixIn = matrix[:maxSize,0]
    matrixParameters = matrix[:maxSize,1]
    #matrixParameters = [matrixParameters[i]*0.1 for i in range(len(matrixParameters))]
    matrixOut = matrix[:maxSize,2]
    reshapedInput  = []
    reshapedOutput =[]
    reshapedGain =[]
    nbExample = len(matrixIn)-num_step
    for i in range(nbExample):
        temp_list = []
        temp_list_out = []
        for j in range(num_step):
            temp_list.append(matrixIn[i+j])
        vectorOut = []
        vectorOut.append(matrixOut[i+num_step-1])
        vectorGain = []
        vectorGain.append(matrixParameters[i+num_step-1])
        reshapedInput.append(np.array(temp_list))
        reshapedOutput.append(np.array(vectorOut))
        reshapedGain.append(np.array(vectorGain))
    return reshapedInput, reshapedOutput , reshapedGain


def shuffleMatrix(data):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    return [data[i] for i in shuffled_indices]



