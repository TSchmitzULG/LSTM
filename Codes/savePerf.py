#!/usr/bin/python
# -*- coding: latin-1 -*-
import time
import os
    
def logPerf(infoLog):
    nameLogFile = os.path.join(infoLog["path"],'logExperiment.txt')
    with open(nameLogFile,'w') as mon_fichier:
        mon_fichier.write("mean square error : {} after {} epoch and a duration of {} \n".format(infoLog["MSE"] ,infoLog["num_epoch"] ,infoLog["duration"]))
        mon_fichier.write("Parameters :"+'\n')
        mon_fichier.write("tensorflow version %s" %infoLog["version"]+'\n')
        mon_fichier.write("Model Name %s" %infoLog["nameModel"]+'\n')
        mon_fichier.write("num_epoch %d" %infoLog["num_epoch"]+'\n')
        mon_fichier.write("num_hidden %d" %infoLog["num_hidden"]+'\n')
        mon_fichier.write("num_step %d" %infoLog["num_step"]+'\n')
        mon_fichier.write("Number of layers %d" %infoLog["n_layer"]+'\n')
        mon_fichier.write("batch_size %d" %infoLog["batch_size"]+'\n')
        mon_fichier.write("input length %d" %infoLog["maxSize"]+'\n')
        mon_fichier.write("dropout %f" %infoLog["trainDropout"]+'\n')
        mon_fichier.write("Total num of trainable parameters %d" %infoLog["totalParameters"]+'\n')
    return 
