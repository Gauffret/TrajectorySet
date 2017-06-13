""" 
    @author: Jacob Gildenblat, 2014
    License: you may use this for whatever you like 

    Contributors :
        松井 賢司 2015-2017
        Gwladys Auffret 2017
"""


import sys, glob, argparse
import matplotlib.pyplot as plt
import numpy as np
import math, cv2
from scipy.stats import multivariate_normal
import time
import pickle
import random
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import os

import cython
from cython.parallel import parallel,prange
cimport numpy as np
from cpython cimport bool


def dictionary(descriptors, int N):
    """
       dictionary(descriptors, int N)

       Create a dictionnary of size N with the descriptors given.
       Using OpenCV.
    """
    print("dictionary")
    em = cv2.EM(N)
    print("cv2")
    em.train(descriptors)
    print("em")
    return np.float32(em.getMat("means")), \
        np.float32(em.getMatVector("covs")), np.float32(em.getMat("weights"))[0]


def image_descriptors(char * file):
    """
        image_descriptors(char * file)

        Return descriptors of the given file.
    """
    #print(file)
    descriptors = np.fromfile(file, dtype='f4')
    descriptors = descriptors.reshape((-1,750)) #3*3=9 9*30=270 / sizeOfLine = 750

    return descriptors

 
def folder_descriptors(char * folder):
    """
        folder_descriptors(char * folder)

        Create descriptors of each file of the folder (see image_descriptors function).
        Keep 1% descriptors of each file, choose randomly.
    """
    print(folder)
    files = glob.glob(folder + "/*g01*.txt")

    print("Calculating descriptors. Number of images is", len(files))       
    all_random_feature = np.zeros([1,750],dtype="f4")
    for file in files:
        file_feature = image_descriptors(file) 
        number = range(0,len(file_feature))
        random_number = random.sample(number,int(len(file_feature)*0.01)) #1%        
        if len(random_number) > 1.0:
            random_feature = np.concatenate([file_feature[i:i+1,:] for i in random_number])
            all_random_feature = np.concatenate((all_random_feature,random_feature))
            all_random_feature = all_random_feature[1:,:]
    return all_random_feature

def likelihood_moment(x, ytk, moment):
    """
        likelihood_moment(x, ytk, moment)

        Calculating x to the moment (power), multiply by the probability ytk.      
    """
    x_moment = np.power(np.float32(x), moment) if moment > 0 else np.float32([1])
    return x_moment * ytk

def likelihood_statistics(samples, means, covs, weights):
    """
        likelihood_statistics(samples, means, covs, weights)

        Calculing the statistics of likelihood
    """
    gaussians, s0, s1,s2 = {}, {}, {}, {}
    samples = zip(range(0, len(samples)), samples)
    
    g = [multivariate_normal(mean=means[k], cov=covs[k]) for k in range(0, len(weights)) ]

    for index, x in samples:
        gaussians[index] = np.array([g_k.pdf(x) for g_k in g])     
        gaussians[index][(gaussians[index])>10000000] = 10000000
        gaussians[index][(gaussians[index])<0.00000001] = 0.00000001
    for k in range(0, len(weights)):
        s0[k], s1[k], s2[k] = 0, 0, 0
        for index, x in samples:
            probabilities = np.multiply(gaussians[index], weights)
            probabilities = probabilities / np.sum(probabilities)      
            s0[k] = s0[k] + likelihood_moment(x, probabilities[k], 0)
            s1[k] = s1[k] + likelihood_moment(x, probabilities[k], 1)
            s2[k] = s2[k] + likelihood_moment(x, probabilities[k], 2)

    return s0, s1, s2

def fisher_vector_weights(s0, s1, s2, means, covs, w, T):
    """
        fisher_vector_weights(s0, s1, s2, means, covs, w, T)

        Return float's list of fisher vector weights.
    """
    return np.float32([((s0[k] - T * w[k]) / np.sqrt(w[k]) ) for k in range(0, len(w))])

def fisher_vector_means(s0, s1, s2, means, sigma, w, T):
    """
        fisher_vector_means(s0, s1, s2, means, sigma, w, T)

        Return float's list of fisher vector means.
    """
    return np.float32([(s1[k] - means[k] * s0[k]) / (np.sqrt(w[k] * sigma[k])) for k in range(0, len(w))])

def fisher_vector_sigma(s0, s1, s2, means, sigma, w, T):
    """
        fisher_vector_sigma(s0, s1, s2, means, sigma, w, T)

        Return float's list of fisher vector sigma.
    """
    return np.float32([(s2[k] - 2 * means[k]*s1[k]  + (means[k]*means[k] - sigma[k]) * s0[k]) / (np.sqrt(2*w[k])*sigma[k])  for k in range(0, len(w))])

def normalize(fisher_vector):
    """
        normalize(fisher_vector)

        Normalize the list of float int fisher_vector 
    """
    v = np.sqrt(abs(fisher_vector)) * np.sign(fisher_vector)
    return v / np.sqrt(np.dot(v, v))

def fisher_vector(samples,file_name, group, means, covs, w):
    """
        fisher_vector(samples,file_name, group, means, covs, w)

        Calculate and return the fisher vector
    """
    group.append(int(file_name[-10:-8]))#group divide
        
    s0, s1, s2 =  likelihood_statistics(samples, means, covs, w)
    T = samples.shape[0]
    covs = np.float32([np.diagonal(covs[k]) for k in range(0, covs.shape[0])])
    a = fisher_vector_weights(s0, s1, s2, means, covs, w, T)
    b = fisher_vector_means(s0, s1, s2, means, covs, w, T)
    c = fisher_vector_sigma(s0, s1, s2, means, covs, w, T)
    fv = np.concatenate([np.concatenate(a), np.concatenate(b), np.concatenate(c)])
    fv = normalize(fv)
    return fv

def getWords(char * input_folder, int nbThread):
    """
        getWords(char * input_folder, int nbThread)

        Return the list of descriptors in the sub-folders of input_folder.
        Using OpenMP to parallelize with nbThread threads.
    """
    words = []
    folders = glob.glob(input_folder + '/*')
    cdef int size = len(folders)
    cdef int i
    with  nogil,parallel(num_threads=nbThread):
        for i in prange(size, schedule='static' ):
            with gil:
                words.append(folder_descriptors(folders[i]))
    return words

def generate_gmm(char * input_folder, int N, int nbThread):
    """
        generate_gmm(char * input_folder, int N, int nbThread)

        Creation of GMM (Gaussian Mixture Model) with informations in sub-folders of folder.
        Utilization of N Gaussians.
        nbThread used in getWords function.
    """
    print("generate gmm")
    words = np.concatenate(getWords(input_folder, nbThread)) 
    
    print("Training GMM of size", N)         
    means, covs, weights = dictionary(words, N)
    #Throw away gaussians with weights that are too small:
    th = 1.0 / N
    means = np.float32([m for k,m in zip(range(0, len(weights)), means) if weights[k] > th])
    covs = np.float32([m for k,m in zip(range(0, len(weights)), covs) if weights[k] > th])
    weights = np.float32([m for k,m in zip(range(0, len(weights)), weights) if weights[k] > th])
    print("before_threshold")
    for i in range(0, len(weights)):
         diag = np.diag(covs[i])
         diag_s = diag.copy()
         diag_s.sort()
         if diag_s[0] < 0.00001:
             zis = [j for j, x in enumerate(diag) if x == diag_s[0]]
             for zi in zis:
                 covs[i,zi,zi] = 0.00001
    np.save("../result/Tmeans.gmm", means)
    np.save("../result/Tcovs.gmm", covs)
    np.save("../result/Tweights.gmm", weights)
    return means, covs, weights

def get_fisher_vectors_from_folder(char * folder, gmm, group):
    """
        get_fisher_vectors_from_folder(char * folder, gmm, group)

        Return the list of fisher vectors created by the files in the folder
    """
    files = glob.glob(folder + "/*g01*.txt")
    return np.float32([fisher_vector(image_descriptors(file), file, group, *gmm) for file in files])

def fisher_features(char * folder, int nbThread, group, gmm):
    """
        fisher_features(char * folder, int nbThread, group, gmm)

        Creation of the dictionary of features by folder.
        Using OpenMP to parallelize with nbThread threads.
    """
    features = {}
    folders = glob.glob(folder + "/*")
    cdef int size = len(folders)
    cdef int i
    with  nogil,parallel(num_threads=nbThread):
        for i in prange(size, schedule='static' ):
            with gil:
                print(folders[i])
                features[folders[i]] = get_fisher_vectors_from_folder(folders[i], gmm, group)
    return features

def train(train,group):
    """
        train(train,group)

        Training of the classification
    """
    X = np.concatenate(train.values())
    Y = np.concatenate([np.float32([i]*len(v)) for i,v in zip(range(0, len(train)), train.values())])
    Y_sum = np.zeros([len(Y)])
    for g_number in len(25):
        X_train = np.zeros([1,len(X[0])]) 
        X_test = np.zeros([1,len(X[0])]) 
        Y_train = np.zeros(1) 
        Y_test = np.zeros(1) 
        for index in len(group):
            print(len(group))
            if group(index) == g_number+1 :
                X_test = np.concatenate([X_test,X[index:index+1, :]])
                Y_test = np.concatenate([Y_test,[Y[index]]])

            else:
                X_train = np.concatenate([X_train,X[index:index+1, :]])
                Y_train = np.concatenate([Y_train,[Y[index]]])
             
            clf = svm.SVC(kernel='rbf', gamma =0.1,C = 1000)
            clf.fit(X_train,Y_train)
             
            res = float(sum([a==b for a,b in zip(clf.predict(X_test), Y_test)])) / len(Y)                  
            Y_sum[index] = a

    print classification_report(Y,Y_sum)

    return res
  
def load_gmm(char * folder = ""):
    """
       load_gmm(char * folder = "")

       Load a preexisting gmm saved in folder.
       Need of files  "means.gmm.npy", "covs.gmm.npy" and "weights.gmm.npy"
    """
    print("load pre-existing gmm")
    files = ["Tmeans.gmm.npy", "Tcovs.gmm.npy", "Tweights.gmm.npy"]
    return map(lambda file: np.load(file), map(lambda s : folder + "/" + s , files))

#Main
def main(bool loadgmm=False, int number = 10, int nbThread = 15):
    """
        main(bool loadgmm=False, int number = 10, int nbThread = 15)

        Main function for calculating fisher vectors.
        The boolean loadgmm load a preexisting GMM dictionary when true, else it is created
        The integer number is the number of words in GMM dictionnary
        The integer nbThread is the number of thread for parallelization
    """
    cdef float start = time.time()

    cdef char * path = '../result'
    cdef char * gmm_path ='/media/gwladys/36A831ACA8316C0D/result'

    gmm = load_gmm(path) if loadgmm else generate_gmm(gmm_path, number, nbThread)


    cdef float elapsed_time = time.time() - start
    print ("elapsed_time_gmm:{0}".format(elapsed_time)) + "[sec]"

    group = []

    fisher_feature = fisher_features(gmm_path, nbThread, group, gmm)
    #
    elapsed_time = time.time() - start
    print ("elapsed_time_fisher:{0}".format(elapsed_time)) + "[sec]"

    with open('../result/Tfisher_dict0.pickle','wb') as f:
        pickle.dump(fisher_feature,f)
    with open('../result/Tfisher_group0.txt','wb') as f:
        f.write("\n".join(map(lambda x: str(x), group)) + "\n")
