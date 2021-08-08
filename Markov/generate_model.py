import numpy as np
from markov import StationaryMarkovChain

class MarkovModelGenerator():

    def generate(data:np.ndarray, function = None) -> StationaryMarkovChain:
        """
            TODO:
        """

        mkModel = MarkovModelGenerator.generate_without_normalization(data,function = function)
        mkModel.normalize()
        
        return mkModel


    def generate_without_normalization(data, function = None):
        """
            TODO
        """
        values      = np.unique(data)
        nodes       = len(values)
        iter_mat    = np.zeros((nodes,nodes))        
        indexes     = np.array(list(map(lambda x: 
                np.nonzero(values == x)[0][0],  data)))

        K = []

        if function == None: 
            K = zip(indexes,np.roll(indexes,-1))
        else: 
            K = function(indexes)

        for i, j in K:
            iter_mat[i][j] += 1

        return StationaryMarkovChain(iter_mat, 
                    initial_distribution = {str(val):0 for val in values})