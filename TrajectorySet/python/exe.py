# -*- coding: utf-8 -*-
""" 
    @author Kenji Matsui
    
    This file use DenseTrackStab in the folder release
    If you don't have it, use the Makefile
    At line 16, put the path of your videos and adapt line 21
    The result will appears in the folder result
"""

import os,commands
import subprocess
import glob
import time


start = time.time()

folders = glob.glob("/home/gwladys/Videos/UCF50/*")
for Vfolder in folders:
    foldername = Vfolder[27:] #You keep only the name of the last folder
    files = glob.glob(Vfolder + "/*.avi")
    elapsed_time = time.time() - start
    print ("elapsed_time_{0}:{1}".format(foldername,elapsed_time)) + "[sec]"
    for F in files:
        name = F.split("/")
        filename = name[6]
        filename = filename[:-4]
        print("../release/DenseTrackStab %s ../result/%s/%s.txt \
        " %(F,foldername,filename))
        commands.getoutput("mkdir ../result/%s " %(foldername))
        commands.getoutput("../release/DenseTrackStab %s ../result/%s/%s.txt \
        " %(F,foldername,filename))
