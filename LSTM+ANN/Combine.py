## Combining both the models (ANN + LSTM)

# First load both the models here
# Then make prediction on LSTM model to get SQ of next day
# Put that SQ into the ANN model to get the prediction

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error  
from copy import deepcopy as dc
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")
import pickle
from LSTM import LSTM

with open('ann.pkl', 'rb') as f:
    ann_model = pickle.load(f)

with open('lstm.pkl', 'rb') as f:
    lstm_model = pickle.load(f)

# To be completed
