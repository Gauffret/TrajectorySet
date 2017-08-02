""" 
    @author: Gwladys Auffret, 2017
    License: you may use this for whatever you like 

"""

import fisher
import time
import pickle
import os

def main(loadgmm, number, nbThread):
    """
        main(bool loadgmm=False, int number = 10, int nbThread = 15)

        Main function for calculating fisher vectors.
        The boolean loadgmm load a preexisting GMM dictionary when true, else it is created
        The integer number is the number of words in GMM dictionnary
        The integer nbThread is the number of thread for parallelization
    """
    start = time.time()
    path = '../result/HMDB51'
    gmm_path ='/media/gwladys/1DD3E28B3E6EB2D5/HMDB51'

    #GMM
    gmm = fisher.load_gmm(path) if loadgmm else fisher.generate_gmm(gmm_path, path, number, nbThread)


    elapsed_time = time.time() - start
    print ("elapsed_time_gmm:{0}".format(elapsed_time)) + "[sec]"

    # Fisher Vectors
    

    fisher_feature = fisher.fisher_features(gmm_path, nbThread, gmm)
    elapsed_time = time.time() - start
    print ("elapsed_time_fisher:{0}".format(elapsed_time)) + "[sec]"

    with open('../result/HMDB51/fisher_dict.pickle','wb') as f:
        pickle.dump(fisher_feature,f)

    group = fisher.get_group(gmm_path)
    with open('../result/HMDB51/fisher_group.txt','wb') as f:
        f.write("\n".join(map(lambda x: str(x), group)) + "\n")

main(True,64,10)