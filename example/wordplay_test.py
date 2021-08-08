import sys
sys.path.append('Markov')

import re
import numpy as np

from markov import StationaryMarkovChain
from generate_model import MarkovModelGenerator


data = []
# Read and format data as a list of words split on spaces, 
# dots, commas and other punctuation marks.
with open('./example/dataset/shrekscript.txt', 'r') as reader:
    for line in reader.readlines():
        line = line.lower()
        arr = re.findall(r"[\w']+|[.,!?;]", line)
        data.extend(arr)

# Initialize the model based on linear data.
mkModel = MarkovModelGenerator.generate(np.array(data))
# set to a Dirac distribution to the 12th key value
mkModel.set_to_dirac_distr(list(mkModel.distribution())[12])

smallsentence = []

# Generate a sentence of any size, whenever there's a dot,
# it stops to generate new words.
for distr in mkModel:
    v = np.random.choice(list(distr), p=list(distr.values()))
    smallsentence.append(v)

    mkModel.set_to_dirac_distr(v)
    if v == ".":
        break
    
print(" ".join(smallsentence))



