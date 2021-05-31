
import sys
sys.path.append('Markov')

import multiprocessing

import random
import numpy as np

from markov import MarkovModel 
from generate_model import MarkovModelGenerator


def is_Prime(n):
    """
    Miller-Rabin primality test.
 
    A return value of False means n is certainly not prime. A return value of
    True means n is very likely a prime.

    See https://rosettacode.org/wiki/Miller%E2%80%93Rabin_primality_test#Python
    """
    if n!=int(n):
        return False
    n=int(n)
    #Miller-Rabin test for prime
    if n==0 or n==1 or n==4 or n==6 or n==8 or n==9:
        return False
 
    if n==2 or n==3 or n==5 or n==7:
        return True
    s = 0
    d = n-1
    while d%2==0:
        d>>=1
        s+=1
    assert(2**s * d == n-1)
 
    def trial_composite(a):
        if pow(a, d, n) == 1:
            return False
        for i in range(s):
            if pow(a, 2**i * d, n) == n-1:
                return False
        return True  
 
    for i in range(8):#number of trials 
        a = random.randrange(2, n)
        if trial_composite(a):
            return False
 
    return True  


# Generate a mkModel of fixed shape and 
# then generate based on nodes_values
mkModel = MarkovModel(np.zeros((10,10)), 
        nodes_label = np.array([f'{i}' for i in range(10)]))

maxPrimes = 1000;
iter      = 0;

def gen_with_line(num):
    # Given a fixed model add data points without rolling data.
    k = zip(list(num[:-1]),list(num[1:]))
    for digit1, digit2  in k:
        i = np.nonzero(mkModel.nodes_label == digit1)[0][0]
        j = np.nonzero(mkModel.nodes_label == digit2)[0][0]
        mkModel.iteration_matrix[i][j] += 1
        
    return mkModel.iteration_matrix

po = multiprocessing.Pool(4)


with open("./example/dataset/list_primes.txt", "r") as reader: 
    for line in reader.readlines():
        
        line = line.split("\t")
        line[-1] = line[-1].split("\n")[0]
        p = po.map(gen_with_line, line)
        mkModel.iteration_matrix = sum(p)
        iter += 1
        if iter > maxPrimes:
            break

mkModel.normalize()


def generate_random_prime(n):
    arr = []
    digits = [f'{i}' for i in range(10) if i % 2 == 1]
    for i in range(n):
        k = random.choice(digits)
        arr.append(k)
    return int("".join(arr))

# Generate an example of a prime with 10 digits
actually_prime_markov = 0
actually_prime_random = 0
N = 1000
lengthPrime = 100
for i in range(N):
    primeMarkov = mkModel.evaluate_path_of_n_steps(lengthPrime)
    primeMarkov = mkModel.path_to_label_path(primeMarkov)
    primeMarkov = int("".join(primeMarkov))
    b  = is_Prime(primeMarkov)
    primeRandom = generate_random_prime(lengthPrime)
    c = is_Prime(primeRandom)
    print("Number %a was generated, it is a prime: %a " % (primeMarkov, b))
    if b:
        actually_prime_markov += 1 
    if c:
        actually_prime_random +=1

print("In total we get %a probability of prime number genrated via Markov chain" % (actually_prime_markov/N))
print("In total we get %a probability of prime number genrated at random" % (actually_prime_random/N))