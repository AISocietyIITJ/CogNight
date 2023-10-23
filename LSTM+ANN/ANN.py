import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error  
from copy import deepcopy as dc
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

################
# READING FILE #
################

dfml = pd.read_csv('Sleep_Efficiency.csv')
dfml = dfml.drop(['Bedtime', 'Wakeup time', 'ID','REM sleep percentage','Deep sleep percentage','Light sleep percentage','Awakenings'] ,axis=1)
dfml['Smoking status'] = dfml['Smoking status'].map({'Yes':1 ,'No':0})
dfml['Gender'] = dfml['Gender'].map({'Male':1 ,'Female':0})
cols = [i for i in dfml.columns if i not in ["Sleep efficiency","Sleep duration"]]
dfml = dfml.dropna()
for col in cols:
  dfml[col] = dfml[col].astype(int)
dfml = dfml.reset_index().drop('index',axis=1)
dfml['Caffeine consumption'] = (dfml['Caffeine consumption']/25).astype(int)

#################
# NORMALIZATION #
#################

dfml['Age'] = (dfml['Age'].max()-dfml['Age'])/(dfml['Age'].max()-dfml['Age'].min())

# Need to improve on normalization technique


####################
# TEST/TRAIN SPLIT #
####################

x_train_2 = dfml[ :int(len(dfml)*0.9)].drop('Sleep duration',axis=1)
x_test_2 = dfml[int(len(dfml)*0.9): ].reset_index().drop(['index','Sleep duration'],axis=1)
y_train_2 = dfml['Sleep duration'][ :int(len(dfml)*0.9)].reset_index().drop(['index'],axis=1)
y_test_2 = dfml['Sleep duration'][int(len(dfml)*0.9): ].reset_index().drop(['index'],axis=1)

#########
# MODEL #     ## NEED TO SWITCH TO PYTORCH ##
#########

from sklearn.neural_network import MLPRegressor
import sklearn
from sklearn import metrics

mlpr = MLPRegressor(hidden_layer_sizes=(1,),solver='adam',learning_rate_init=0.1,random_state=25)
mlpr.fit(x_test_2,y_test_2)

y_pred_whole = mlpr.predict(x_test_2)
ame = sklearn.metrics.mean_absolute_error(y_test_2, y_pred_whole)

#################
# SAVING PARAMS #
#################

import pickle

with open('ann.pkl','wb') as f:
  pickle.dump(mlpr,f)

################
# TESTING EVAL #
################

print(ame)