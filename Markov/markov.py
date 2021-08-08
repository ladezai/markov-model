import numpy as np
from math import isclose 

class StationaryMarkovChain():
    """
        It provides an implementation of stationary finite
        Markov chain `(l, P)` over the set of labels `S` 
        (or nodes of the Markov chain). 

        The following implementation emphasises that a Markov 
        chain represents the evolution of a distribution over 
        time, via a stocastic process determined by `P`. 
        Therefore from an iterable point of view the class is 
        only its distribution, i.e. of type``dict[str,float]``.
        And ``__next__`` method is replaced by ``evaluate_next``. 
        In such a way that allows to easily simulate a path over 
        itself via for-loops.
        
        See ``../examples/word_generator.py`` or 
        ``../examples/wordplay_test.py``.

        Example
        -------
        We represent a trivial markov chain via the following graph:

         ._0.5_             .__1__
        |     |            |     |
        '--> a ----0.5---> b <--'

        then the set S = {'a', 'b'} and the stocastic matrix P
        is the following 
        P = [[0.5,0.5],[0, 1]]

        In terms of the implementation below the set S
        represents the keys of the initial distribution, while
        P is the iteration_matrix. Since no initial distribution
        is given in our example we could consider the Dirac 
        distribution which is 1 in 'a' and 0 in 'b', therefore
        initial_distribution = {'a' : 1.0, 'b': 0.0}.


        Attributes
        ----------
        iteration_matrix : numpy.array
            A stocastic matrix which represents links 
            between each node in the Markov Chain's state.
        current_distr    : dict[str,float]
            A dictionary that provides an initial distribution
            for the MarkovChain. Where keys are used
            as label for each node of the Markov Chain's state.

        Methods
        --------
        set_distr : dict[str,float] -> bool -> None
            Assigns a new value to current_distribution. 
            Checks whether the new value is also a distribution.
        reset_distr : None
    """


    iteration_matrix = np.array([[]])
    current_distr    = dict()
    
    def __init__(self, iteration_matrix:np.array, initial_distribution:dict[str, float]):
        """
            Parameters
            ----------
            iteration_matrix : numpy.array
                The Markov Chain's stocastic matrix, it has to be a 
                square matrix with normalized rows.  
            initial_distribution : dict[str, float]
                A dictionary which provides an initial distribution.
                Note that it must have same length as the rows/columns 
                of the iteration_matrix.

            Raises
            ------
            ValueError
                If initial_distribution and iterations_matrix aren't 
                of same length.
        """
        if iteration_matrix.shape[0] != iteration_matrix.shape[1]:
            raise ValueError("Iteration matrix must be a square-matrix")

        if len(initial_distribution) != iteration_matrix.shape[0]:
            raise ValueError(("Initial distribution and iteration matrix does" +
                             "not have same size."))
        
        self.iteration_matrix = iteration_matrix
        self.current_distr    = initial_distribution
        


    ###########################################################################
    ################## Get, set and update distribution #######################
    ###########################################################################

    def set_distr(self, distr:dict[str,float], checks : bool = False):
        """
            Parameters
            ----------
            distr : dict[str, float]
                A dictionary which provides a distribution.
                Note that it must have same length as the rows/columns 
                of the iteration_matrix.
            checks : bool 
                It provides a way to check whether the distr parameter 
                is a distribution (default is False).

            Raises
            ------
            ValueError
                If initial_distribution and iterations_matrix aren't 
                of same length.
            ValueError 
                If distr is not normalized.
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
            new_values  = self.iteration_matrix.T.dot(values) 
            
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

    