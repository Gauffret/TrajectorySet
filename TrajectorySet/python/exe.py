# -*- coding: utf-8 -*-
""" 
    @author Kenji Matsui
    
    This file use DenseTrackStab in the folder release
    If you don't have it, use the Makefile
    At line 16, put the path of your videos and adapt line 21
    The result will appears in the folder result
"""

import os,commands
import sys
import subprocess
import glob
import time


start = time.time()
folders = glob.glob("/home/gwladys/Videos/UCF-101/*")
for Vfolder in folders[94:95]:
    foldername = Vfolder[29:] #You keep only the name of the last folder
    files = glob.glob(Vfolder + "/*.avi")
    elapsed_time = time.time() - start
    print ("elapsed_time_{0}:{1}".format(foldername,elapsed_time)) + "[sec]"
    for F in files[141:]:
        name = F.split("/")
        filename = name[6]
        filename = filename[:-4]
        commands.getoutput("mkdir /media/gwladys/1DD3E28B3E6EB2D5/result/%s " %(foldername))
        commands.getoutput("../release/DenseTrackStab %s /media/gwladys/1DD3E28B3E6EB2D5/result/%s/%s.txt \
        " %(F,foldername,filename))
