### Markov chains

This library is a small implementation of a Markov chain model.

Some errors may occur in case of a bad conditioned matrix representing 
a markov chain. Errors are given by conditioning of the dataset, to 
prevent it, one can re-normalize the iteration matrix of the 
Markov chain (hoping errors don't add up).

### How to use 

It requires `numpy`. Here's a small example, source code is at 
`/example/`.

```python3

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
```

Which displays something like:

```
they judge people before the heimlich ? uh , you know oh , i'm good look down ! la la la la la la la , orge . i have the back . . if anyone know ! what am rescuing , wow . i was ask . . you're gonna do i was just now come on my gratitude . whistles donkey on . i can nearly see so damn thing you know ? i am authorized to you got instincts . hey , no , dah , i'm on the dragon . ha now you're an orge in
```

### Further development

1. Store a MarkovModel via a JSON file.