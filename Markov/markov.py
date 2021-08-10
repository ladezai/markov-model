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
        Use pickle for serialization since no JSON 
        serialization has been implemented yet.


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
            Assigns a new value to current_distr. 
            Checks whether the new value is also a distribution.
        set_to_dirac_distr : str -> None
            Assigns `1` to the key value given and `0` to all 
            other values of the distribution.
        distribution : dict[str,float]
            Returns the current value of the Markov's chain 
            distribution.
        evaluate_next : int -> None
            Evaluates the development of the distribution after
            `n` steps. 
        normalize : None
            Normalizes rows of the iteration_matrix.
            It's useful in case of some floating point errors.
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
            Sets the current distribution of the Markov Chain. 

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


    def set_to_dirac_distr(self, node : str) -> None:
        """
            Sets to a Dirac distribution centered in the node given.

            Parameters:
            -----------
            node: str 
                A String that represents a label of the Markov Chain.
                It serves as a key in current_distr's dictionary.

            Raises:
            ---------
            KeyError
                If node is not a key in current_distr.
        """
        
        distr = {k:0 for k in self.current_distr}
        distr[node] = 1
        
        self.set_distr(distr, checks=False)
        
        return None
    
    def distribution(self) -> dict[str,float]:
        """
            Gives a view of current_distr.

            Returns:
            --------
            dict[str,float]     
        """
        return self.current_distr

    
    ###########################################################################
    ############################# Simulation ##################################
    ###########################################################################

    def evaluate_next(self, n : int = 1, checks : bool = False) -> None:
        """
            Evaluates the distribution after n-steps in-place.
            In mathematical terms it evalues l * P^n, where l is the 
            initial_distr, and P the iteration_matrix.

            Parameters:
            ------------
            n : int
                Represents the number of steps to evaluate 
                on the Markov chain. Therefore it has to be 
                non-negative (default value is 1).
            checks: bool 
                If checks is True, it checks whether the current_distr
                is still a distribution at the end of the simulation,
                and if a negative number of step is inputed 
                (default value is False).

            Raises:
            --------
            ValueError
                If n is negative and check is True.
            ValueError
                If the simulation generates an invalid distribution.
        """
        
        if checks:
            if n < 0:
                raise ValueError("Can't simulate a negative number of steps")

        distr = self.distribution()
        for i in range(n):
            values      = np.array(list(distr.values()))
            new_values  = self.iteration_matrix.T.dot(values) 
            
            distr = {key:new_values[j] 
                for j,key in enumerate(self.current_distr)}
            
        self.set_distr(distr,checks=checks)

        return None

    def normalize(self) -> None:
        """
            Normalizes by rows the iteration_matrix in-place.
        """
        l = len(self)
        for i in range(l):
            N = sum(self.iteration_matrix[i])
            self.iteration_matrix[i] /= N

        return None

    def __iter__(self):
        return self

    def __next__(self):
        self.evaluate_next(n=1)
        return self.current_distr

    def __del__(self):
        del self.current_distr
        del self.iteration_matrix

    def __len__(self):
        return len(self.current_distr)

    def __str__(self):
        return ("""It's a Markov chain (l, P), where
                     P = """ + str(self.iteration_matrix) + """
                     l = """ + str(self.current_distr))

    