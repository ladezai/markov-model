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
            TODO
        """
        self.iteration_matrix = iteration_matrix
        self.current_distr    = distribution


    ###########################################################################
    ################## Get, set and update distribution #######################
    ###########################################################################

    def set_current_distr(self, distr:dict[str,float], checks : bool = False):
        """
            TODO
        """
        if checks:
            M = sum(list(distr.values()))
            if not isclose(M,1,abs_tol=0.0001):
                raise ValueError("Dictionary given is not a distribution.")

            if len(self) != len(distr):
                raise ValueError("Distributions have different length.")

        self.current_distr = distr
        return None

    def reset_distr(self) -> None:
        """
            TODO
        """
        self.current_distr = {k:0 for k in self.current_distr}
        return None

    def update_distr(self, node:str, value:float) -> None:
        """
            TODO
        """
        self.current_distr[node] = value
        
        return None

    def set_to_dirac_distr(self, node:str) -> None:
        """
            TODO
        """
        self.reset_distr()
        self.current_distr[node] = 1

        return None
    
    def distribution(self) -> dict[str,float]:
        return self.current_distr

    
    ###########################################################################
    ###########################################################################
    ###########################################################################

    def evaluate_next(self, n : int = 1, normalize : bool = True) -> None:
        """
            Evaluate in-place the new distribution after 1 step.
        """
        for i in range(n):
            values      = np.array(list(self.current_distr.values()))
            new_values  = self.iteration_matrix.dot(values) 
            
            i = 0
            for key in self.current_distr:
                self.current_distr[key] = new_values[i]
                i += 1 
            
            del i

            # Normalization to a distribution
            if normalize:
                self.normalize_distr()

        return None

    def normalize(self) -> None:
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

    def normalize_distr(self) -> None:
        """
            TODO
        """
        K = sum(list(self.current_distr.values()))
        for key in self.current_distr:
            self.current_distr[key] /= K

        return None
    
    ###########################################################################
    ########################### Theoretical stuff #############################
    ###########################################################################

    def is_irreducible(self) -> bool:
        """
            Returns `True` if the Markov chain is irreducible.

            It runs in about O(n^5) where n * n is the size of 
            the iteration matrix.
        """

        l = len(self)
        i0   = -1
        j0   = -1
        
        # Note that extrema can be removed since in either 
        # the rows can't be normalized if they were all 0s or
        # the square has length 1
        for i in range(l-1):
            for j in range(1,l-1):
                # Check whether the entire row i, from j onwards is null
                if not np.any(self.iteration_matrix[i,j:]):
                    i0 = i
                    j0 = j

                # If a possible square is not found return at the beginning
                # of the loop. 
                if i0 == -1:
                    continue

                # length of the square of 0s
                M = l - j0

                # Check whether M <= 1, therefore if the square is of trivial 
                # size and check whether the square could have 
                # the same height as the width.                
                if M <= 1 or i0+M > l:
                    i0 = -1
                    j0 = -1
                    continue

                # If the entire square is 0s then there's a closed set,
                # therefore we can conclude is not irreducible.
                if not np.any(self.iteration_matrix[i0:(i0+M),j0:]):
                    return False
                else: 
                    i0 = -1
                    j0 = -1
            
        return True

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

    