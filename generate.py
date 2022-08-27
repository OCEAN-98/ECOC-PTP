import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import pandas as pd
import os
from sklearn.utils import shuffle
from itertools import product

data = pd.read_csv('/Users/Ocean/Documents/Git/ECOC-PTP/data_0822_total_1.csv')
data = shuffle(data)
data.to_csv('/Users/Ocean/Documents/Git/ECOC-PTP/data_shuffle_1.csv')