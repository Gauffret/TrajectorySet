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
num = sys.argv[1]
folders = glob.glob("/home/gwladys/Videos/UCF50/"+num+"/*")
for Vfolder in folders:
    foldername = Vfolder[29:] #You keep only the name of the last folder
    files = glob.glob(Vfolder + "/*.avi")
    elapsed_time = time.time() - start
    print ("elapsed_time_{0}:{1}".format(foldername,elapsed_time)) + "[sec]"
    for F in files:
        name = F.split("/")
        filename = name[7]
        filename = filename[:-4]
        print("../release/DenseTrackStab %s ../result/%s/%s.txt \
        " %(F,foldername,filename))
        commands.getoutput("mkdir /media/gwladys/36A831ACA8316C0D/result/%s " %(foldername))
        commands.getoutput("../release/DenseTrackStab %s /media/gwladys/36A831ACA8316C0D/result/%s/%s.txt \
        " %(F,foldername,filename))
