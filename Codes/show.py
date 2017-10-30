#!/usr/bin/python
# -*- coding: latin-1 -*-
"""function to load image from picle """
import pickle
import matplotlib.pyplot as plt
import sys

def showPickle(name):
	ax = pickle.load(open(name,'rb'))
	plt.show()
	#input("Press Enter to continue...")

if __name__ == '__main__':
	print ('sys.argv: ', sys.argv)
	if len(sys.argv) > 1:
		showPickle(sys.argv[1])
	else:
		print ("no argument")
