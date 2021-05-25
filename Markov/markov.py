import numpy as np
from numpy.random import choice

class MarkovModel():
    """
        A Markov model is a data structure composed of 3 elements:
            1. `MarkovModel.iteration_matrix` which is a 2D numpy array 
                of shape `n x n` (i.e. matrix) iteration matrix of the 
                discrete markov model. 
            2. `MarkovModel.nodes_label` is an array to substitute 
                each of the `n` nodes with a label. 
            3. `MarkovModel.current_state` which represents the point
                at which the simulation is at.
        
        In order to generate a Markov Model one has to call a 
        constructor method, which gives an `iteration_matrix` 
        (a n x n numpy array), `nodes_label` has to have length 
        `n` and is facoltative,
        `initial_state` is also facoltative and is an integer between
        `0` and `n-1`.     
    """
    iteration_matrix = np.array([[]])
    nodes_label      = np.array([])
    rows             = 0
    current_state    = 0
    
    
    def __init__(self, iteration_matrix, nodes_label=None, initial_state=0):
        self.iteration_matrix = iteration_matrix
        self.nodes_label      = nodes_label
        self.current_state    = initial_state
        self.rows             = self.iteration_matrix.shape[0]
    

    def set_current_state(self, state):
        """
            Set current `state` based on nodes' labels.

            Example:
            ```
                > mm = MarkovModel([[1,0][0,1]], nodes_label=["pizza","soup"])
                > mm.set_current_state("pizza")
                > mm.current_state
                  0
        """
        if self.nodes_label.shape[0] < 1:
            raise NameError("There are no labels saved in the Markov Model")
            return

        if not state in self.nodes_label:
            raise NameError("The state isn't in MarkovModel.nodes_label")
            return

        self.current_state = np.nonzero(self.nodes_label == state)[0][0]
        return self.current_state

     
    def evaluate(iteration_matrix, choices, initial_state):
        """
            A static version of `evaluate_next`.
        """
        return choice(choices, 1, p=iteration_matrix[initial_state])[0]


    def evaluate_next(self):
        """
            Assume a Markov model that has probability $1$
            that is in $u_0 \in \{0, \dots, n-1\}$.
            And further assume an iteration matrix $A$ with 
            dimensions $n \times n$. 
            It evaluates, based on the distribution given 
            by $M \cdot x_0$, which of the $\{0, \dots, n-1\}$
            state should follow. 
        """
        self.current_state = MarkovModel.evaluate(self.iteration_matrix, 
                            self.rows,
                            self.current_state)
        return self.current_state
   

    def evaluate_after_n(self,n):
        """1
            Basically the same as `evaluate_next` although 
            the iteration matrix is given by `A^n`.
        """
        self.current_state = MarkovModel.evaluate(self.iteration_matrix**n, 
                                        self.rows, 
                                        self.current_state)
        return self.current_state


    def evaluate_path_of_n_steps(self,n):
        """
            Evaluates a path of n steps on the Markov Model and 
            returns an array of the nodes passed bys.
        """
        
        iters = np.zeros(n)
        for i in range(n):
            iters[i] = self.evaluate_next()
        
        return iters

    def state_to_label(self,state):
        """
            Requires an integer `state' between `0` and 
            `MarkovModel.nodes_label`'s  length. 
        """

        # Since numpy stores efficiently floating point
        # and most times I just use numpy's arrays, 
        # I have to round the floating points. 
        state = round(state)

        if state > self.rows or state < 0: 
            raise NameError("state given is higher the maximum  \
                            value or lower than minimum value")
            return 

        return self.nodes_label[state]

    def path_to_label_path(self,path):
        """
            Given an array of integers, `0` and 
            `self.nodes_label`'s  length,  
            returns an array of `self.nodes_labels`. 
        """
        return np.array(list(map(self.state_to_label,path)))

    def renormalize(self):
        """
            Normalization per rows of `self.iteration matrix` 
            based on the current `self.iteration_matrix`.

            Note: It's damaging repeating this function 
            over-and-over again. 
        """
        for i in range(self.rows):
            N = sum(self.iteration_matrix[i])
            self.iteration_matrix[i] /= N

    
    def __str__(self):
        return str(self.iteration_matrix) + str(self.nodes_label)