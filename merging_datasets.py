from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from api import State, util
import pickle
import os.path
from argparse import ArgumentParser
import time
import sys
import numpy as np
# This package contains various machine learning algorithms
import sklearn
import sklearn.linear_model
from sklearn.neural_network import MLPClassifier  ###########
from sklearn.linear_model import LogisticRegression  ###########
import joblib
from scipy.stats import bernoulli #### added

from bots.rand import rand
from bots.rdeep import rdeep
from bots.bully import bully
from bots.kbbot import kbbot
from bots.ml import ml
from bots.ml import ml_evo  # This is how we use our ml_evo bot/py.file
from bots.alphabeta import alphabeta

import operator
import random

# import math

# from deap import base
# from deap import benchmarks
# from deap import creator
# from deap import tools

# from bots.ml.ml import features

with open('/Users/xxx/Desktop/josh_schnapsen_getting_results/bully_dataset.pkl', 'rb') as output: ##### REMEMBER TO CHANGE THIS NAME
    data, target = pickle.load(output) # 13k
print(np.shape(data))
with open('/Users/xxx/Desktop/josh_schnapsen_getting_results/rand_dataset.pkl', 'rb') as output2: ##### REMEMBER TO CHANGE THIS NAME
    data2, target2 = pickle.load(output2) # len: 16k


with open('/Users/xxx/Desktop/schnapsen/ml_dataset.pkl', 'rb') as output3: ##### REMEMBER TO CHANGE THIS NAME
    data3, target3 = pickle.load(output3) # 15k


with open('/Users/xxx/Desktop/schnapsen/rdeep_dataset.pkl', 'rb') as output4: ##### REMEMBER TO CHANGE THIS NAME
    data4, target4 = pickle.load(output4) #15k

data.extend(data2)
target.extend(target2)
data.extend(data3)
target.extend(target3)
data.extend(data4)
target.extend(target4)

with open('/Users/xxx/Desktop/schnapsen/complete_dataset.pkl', 'wb') as output: # Change to your own path
    pickle.dump((data, target), output, pickle.HIGHEST_PROTOCOL)

# with open('/Users/emmastahlberg/Desktop/josh_schnapsen_getting_results/bots/ml/complete_dataset.pkl', 'rb') as output:
#     data, target = pickle.load(output)

# print('Emma')


### Now run the trainmlbot with the completedataset so it creates a model, and then run evoemma with that complete pkl set and model