import glob
import os
import numpy as np
import pickle

files = glob.glob("/media/gwladys/1DD3E28B3E6EB2D5/HMDBtestTrainMulti_7030_splits/*1.txt")

dict = {}

for file in files :
	print file[63:]
	file1 = open(file,"r").read().split("\n")
	base = file[:-5]
	file2 = open(base+"2.txt","r").read().split("\n")
	file3 = open(base+"3.txt","r").read().split("\n")

	for i in range(len(file1)-1):
		dict[file1[i][:-7]] = [file1[i][-2],file2[i][-2],file3[i][-2]]


with open('../result/HMDB51/TrainTestSets.pickle','wb') as f:
	pickle.dump(dict,f)