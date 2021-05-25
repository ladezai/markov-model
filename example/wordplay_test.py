import sys
sys.path.append('Markov')

import re
import numpy as np

from markov import MarkovModel
from generate_model import MarkovModelGenerator


data = []

with open('./example/dataset/shrekscript.txt', 'r') as reader:
    for line in reader.readlines():
        line = line.lower()
        arr = re.findall(r"[\w']+|[.,!?;]", line)
        data.extend(arr)

mkModel = MarkovModelGenerator.generate(np.array(data))
smallsentence = mkModel.evaluate_path_of_n_steps(100)
smallsentence = mkModel.path_to_label_path(smallsentence)
print(" ".join(smallsentence))



