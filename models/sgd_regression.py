import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import warnings
import io
warnings.filterwarnings("ignore")


df = pd.read_csv('sleepdata_2.csv')
df = df.drop(['Start','End','Movements per hour'],axis=1)

df['Sub'] = df['Time asleep (seconds)']-df['SQ*in_bed']
df = df[df['Time asleep (seconds)']>10000]
df['Sleep Quality'] = df['Sleep Quality'].str.rstrip('%').astype('float') / 100.0
df['Time Debt'] = np.zeros(len(df))

for i in range(len(df)-7):
  df['Time Debt'][i] = 176400-df['Time asleep (seconds)'][i+1:i+8].sum()

df['Time Debt'].mean()


X_test = df[['Sleep Quality','Steps']][:150]
Y_test = df[:150]['Sub']
X_train = df[['Sleep Quality','Steps']][150:].reset_index().drop(['index'],axis = 1)
Y_train = df[150:]['Sub'].reset_index().drop(['index'],axis = 1)

fig, ax = plt.subplots(1,2)
ax[1].scatter(X_train['Sleep Quality'],Y_train)
ax[0].scatter(X_train['Steps'],Y_train,c='r')
ax[0].set_xlabel('Steps')
ax[1].set_xlabel('SQ')
ax[0].set_ylabel('Sub')
plt.show()

#normalizing data
print(X_train.mean())
X_norm = (X_train - X_train.mean())/X_train.mean()
X_test_norm = (X_test - X_test.mean())/X_test.mean()
X_test_norm

sgdr = SGDRegressor(max_iter=10000)
sgdr.fit(X_norm,Y_train)

b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(w_norm,b_norm)

y_pred = sgdr.predict(X_test_norm)
print(Y_test[:5])
print(y_pred[:5])

fig, ax = plt.subplots(1,2)
ax[1].scatter(X_test['Sleep Quality'],Y_test,c='r',label='target')
ax[1].scatter(X_test['Sleep Quality'],y_pred,c='b',label='prediction')
ax[0].scatter(X_test['Steps'],Y_test,c='r')
ax[0].scatter(X_test['Steps'],y_pred,c='b')
ax[0].set_xlabel('Steps')
ax[1].set_xlabel('SQ')
ax[0].set_ylabel('Sub')
plt.legend()
plt.show()

sq = float(input("Sleep Quality:"))
steps = int(input("Steps:"))

x_input = np.array([sq,steps]).reshape(-1,2)
x_input_norm = np.zeros((2,2))
x_input_norm[0,0] = (x_input[0,0]-0.782244)/0.782244
x_input_norm[0,1] = (x_input[0,1]-5358.262467)/5358.262467

time_in_bed = (25000-sgdr.predict(x_input_norm)[0])/x_input[0,0]
print("You need to sleep for:", round(time_in_bed/3600,2), "hours")


