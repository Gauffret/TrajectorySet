import matplotlib.pyplot as plt
from pylab import *
import numpy as np

path = "../result/means.gmm.npy"
means = np.load(path)

cellsize = 10
coll = means[:,0].size


for l in range(0,coll):
    
    Cx=[]
    Cy=[]
    
    for i in range(0,375):
        Cx.append(means[l,i*2])
        Cy.append(means[l,i*2+1])
    n = 0
    for i in range(0,5):#y
        for j in range(0,5):#x
            means_x = j*cellsize+(cellsize/2)
            means_y = i*cellsize+(cellsize/2)
            for k in range(0,15):
                means_x = Cx[n]+means_x
                means_y = Cy[n]+means_y
                n = n+1
                plt.plot(means_x,means_y,'r-o',color=cm.hsv(float(i+j)/10.0),ms=4)
    plt.grid()    
    plt.xlim([0,55])
    plt.ylim([0,55]) 
    plt.savefig("../result/plot/means%d.pdf"%l) 
    plt.show()
