import pickle
import random
from datetime import datetime
import collections

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Model

from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat
from deepmatch.models import *
from deepmatch.utils import sampledsoftmaxloss

from annoy import AnnoyIndex