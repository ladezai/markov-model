import numpy as np
from markov import MarkovModel 

class MarkovModelGenerator():

    def generate(data):
        """
           `generate` returns a MarkovModel based on data
            as an indexed array. 
            
        """

        values      = np.unique(data)
        nodes       = len(values)
        iter_mat    = np.zeros((nodes,nodes))        
        indexes     = np.array(list(map(lambda x: 
                np.nonzero(values == x)[0][0],  data)))

        for i, j in zip(indexes,np.roll(indexes,-1)):
            iter_mat[i][j] += 1
        
        # Normalization
        for i, v in enumerate(values):
            occ = np.count_nonzero(data==v)
            iter_mat[i] /= occ
        
        return MarkovModel(iter_mat, nodes_label = values)