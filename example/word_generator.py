import sys
sys.path.append('Markov')

import re
import numpy as np

from markov import StationaryMarkovChain
from generate_model import MarkovModelGenerator

data = []

# Read linear data from a large file and format in lowercase.
with open('./example/dataset/shrekscript.txt', 'r') as reader:
    for line in reader.readlines():
        line = line.lower()
        data.extend(list(line))

# Initialize the Markov chain based on data
mkModel = MarkovModelGenerator.generate(np.array(data))
# Set distribution to a dirac distribution.
mkModel.set_to_dirac_distr(list(mkModel.current_distr)[5])

word = []

# Loops through the markov model until a space or a newline 
# char is found. 
for distr in mkModel:
    q = np.random.choice(list(distr), p=list(distr.values())) 
    if q == " " or q == "\n":
        break

    word.append(q)
    mkModel.set_to_dirac_distr(q)
    
print("".join(word))
