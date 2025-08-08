import numpy
import pickle


for starting_stepsize in [0.1,1.0,10.0]:
    for rie_alg in ["rbfgs"]:
        if rie_alg == "rbfgs":
            for memory in [1,5,10, 15]: