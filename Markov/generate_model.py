import numpy as np
from markov import MarkovModel 
from multiprocessing import Pool

class MarkovModelGenerator():

    def generate(data,roll=True):
        """
           `generate` returns a MarkovModel based on data
            as an indexed array. 
        """

        mkModel = MarkovModelGenerator.generate_without_normalization(data,roll = roll)
        mkModel.normalize()
        
        return mkModel


    def generate_without_normalization(data, roll=True):
        values      = np.unique(data)
        nodes       = len(values)
        iter_mat    = np.zeros((nodes,nodes))        
        indexes     = np.array(list(map(lambda x: 
                np.nonzero(values == x)[0][0],  data)))

        K = []
        if roll: 
            K = zip(indexes,np.roll(indexes,-1))
        else: 
            K = zip(indexes[:-1], indexes[1:])

        for i, j in K:
            iter_mat[i][j] += 1

        return MarkovModel(iter_mat, nodes_label = values)