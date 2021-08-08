import numpy as np
from numpy.random import choice
from math import isclose 

class MarkovModel():
    """
        TODO: add documentation
    """
    iteration_matrix = np.array([[]])
    current_distr    = dict()
    
    def __init__(self, iteration_matrix:np.array, distribution:dict[str, float]):
        """
            distribution : Dict  
        """
        self.iteration_matrix = iteration_matrix
        self.current_distr    = distribution

    def set_current_distr(self, distr:dict[str,float], checks=False):
        """
            
        """
        if checks:
            M = sum(list(distr.values()))
            if not isclose(M,1,abs_tol=0.0001):
                raise ValueError("Dictionary given is not a distribution.")

            if len(self) != len(distr):
                raise ValueError("Distributions have different length.")

        self.current_distr = distr
        return None

    def reset_distr(self):
        ""
        self.current_distr = {k:0 for k in self.current_distr}
        return None

    def update_distr(self, node:str, value:float):
        """
        
        """
        self.current_distr[node] = value
        
        return None

    def set_to_dirac_distr(self, node:str):
        self.reset_distr()
        self.current_distr[node] = 1

        return None

    def evaluate_next(self, n=1, normalize=True):
        """
            Evaluate in-place the new distribution after 1 step.
        """
        while n >= 1:
            values = np.array(list(self.current_distr.values()))
            new_values  = self.iteration_matrix.dot(values) 
            
            i = 0
            for key in self.current_distr:
                self.current_distr[key] = new_values[i]
                i += 1 
            
            del i

            # Normalization to a distribution
            if normalize:
                self.normalize_distr()

            n -= 1
        return None

    def normalize(self):
        """
            Normalization per rows of `self.iteration matrix` 
            based on the current `self.iteration_matrix`.

            Note: It's damaging repeating this function 
            over-and-over again. 
        """
        l = len(self)
        for i in range(l):
            N = sum(self.iteration_matrix[i])
            self.iteration_matrix[i] /= N
        return None

    def normalize_distr(self):
        K = sum(list(self.current_distr.values()))
        for key in self.current_distr:
            self.current_distr[key] /= K

        return None
    
    def distribution(self):
        return self.current_distr
    
    def __iter__(self):
        return self

    def __next__(self):
        self.evaluate_next(n=1,normalize=True)
        return self.current_distr

    def __del__(self):
        del self.current_distr
        del self.iteration_matrix

    def __len__(self):
        return len(self.current_distr)

    def __str__(self):
        return str(self.iteration_matrix) + str(self.current_distr)

    