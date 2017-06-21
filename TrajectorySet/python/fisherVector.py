""" 
    @author: Gwladys Auffret, 2017
    License: you may use this for whatever you like 

"""

import fisher
import time
import pickle
import os

def main(loadgmm=False, number = 64, nbThread = 5):
    """
        main(bool loadgmm=False, int number = 10, int nbThread = 15)

        Main function for calculating fisher vectors.
        The boolean loadgmm load a preexisting GMM dictionary when true, else it is created
        The integer number is the number of words in GMM dictionnary
        The integer nbThread is the number of thread for parallelization
    """
    start = time.time()
    path = '../result'
    gmm_path ='/media/gwladys/36A831ACA8316C0D/result'

    #GMM
    gmm = fisher.load_gmm(path) if loadgmm else fisher.generate_gmm(gmm_path, number, nbThread)


    elapsed_time = time.time() - start
    print ("elapsed_time_gmm:{0}".format(elapsed_time)) + "[sec]"

    # Fisher Vectors
    group = []

    fisher_feature,group = fisher.fisher_features(gmm_path, nbThread, group, gmm)
    print(group)
    elapsed_time = time.time() - start
    print ("elapsed_time_fisher:{0}".format(elapsed_time)) + "[sec]"

    with open('../result/10fisher_dict.pickle','wb') as f:
        pickle.dump(fisher_feature,f)
    with open('../result/10fisher_group.txt','wb') as f:
        f.write("\n".join(map(lambda x: str(x), group)) + "\n")

main(True,64,10)