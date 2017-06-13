# -*- coding: utf-8 -*-
"""
	@version 13/06/2017

	@author: Gwladys Auffret
"""

import matplotlib.pyplot as plt
from pylab import *
import numpy as np
import os
import glob



def plotFile(file, resultFolder="") :

	means = np.fromfile(file,dtype ="f4")
	means = means.reshape((-1,750))
	cellsize = 10
	colSize = means[:,0].size

	print(file,colSize)
	for l in range(0,10):
	    
	    Cx=[]
	    Cy=[]
	    
	    for m in range(0,375):
	        Cx.append(means[l,m*2])
	        Cy.append(means[l,m*2+1])
	    n = 0
	    for i in range(0,5):#y
	        for j in range(0,5):#X
	            means_x = []
	            means_y = []
	            means_x.append(j*cellsize+(cellsize/2))
	            means_y.append(i*cellsize+(cellsize/2))
	            for k in range(0,15):
	                means_x.append(Cx[n]+means_x[k])
	                means_y.append(Cy[n]+means_y[k])
	                n = n+1
	            plt.plot(means_x,means_y,'r-o',color=cm.hsv(float(i+j)/10.0),ms=4)
	    plt.grid()    
	    plt.xlim([0,55])
	    plt.ylim([0,55]) 
	    plt.savefig(resultFolder+"/means%d.pdf"%l) 
	    #plt.show()



def plot(inputFolder="/media/gwladys/36A831ACA8316C0D/result") : 
	folders = glob.glob(inputFolder + '/*')
	for folder in folders :
		folderName = folder.split("/")[5]
		os.mkdir('../result/plot/'+folderName)
		files = glob.glob(folder + '/*')
		
		#for file in files :
		fileName = files[0].split("/")[6]
		resultFolder = '../result/plot/'+folderName+'/'+fileName
		os.mkdir(resultFolder)
		plotFile(files[0], resultFolder)

plot()