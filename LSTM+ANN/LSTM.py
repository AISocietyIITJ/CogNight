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

df = pd.read_csv('sleepdata_2.csv')
df['Sleep Quality'] = df['Sleep Quality'].str.rstrip('%').astype('float') / 100.0
df = df[df['Sleep Quality']>0.4]
data = df['Sleep Quality'].copy().reset_index().drop('index',axis=1)
df[['Date','Time']] = df['Start'].str.split(expand=True)
data = df[['Date','Sleep Quality']].copy().reset_index().drop('index',axis=1)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#######################
# PREPARING DATAFRAME #
#######################


def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)

    df.set_index('Date', inplace=True)

    for i in range(1, n_steps+1):
        df[f'Sleep Quality(t-{i})'] = df['Sleep Quality'].shift(i)

    df.dropna(inplace=True)

    return df

lookback = 7
shifted_df = prepare_dataframe_for_lstm(data, lookback)
shifted_df_as_np = shifted_df.to_numpy()

#################
# NORMALIZATION #
#################

scaler = MinMaxScaler(feature_range=(-1, 1))
shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)

####################
# TEST/TRAIN SPLIT #
####################

X = shifted_df_as_np[:, 1:]
y = shifted_df_as_np[:, 0]

split_index = int(len(X) * 0.9)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

X_train = X_train.reshape((-1, lookback, 1))
X_test = X_test.reshape((-1, lookback, 1))

y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()

#####################
# PREPARING DATASET #
#####################

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

########################
# PREPARING DATALOADER #
########################

batch_size = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for _, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)

########
# LSTM #
########

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

##################
# TRAIN FUNCTION #
##################

def train_one_epoch(model, train_loader, loss_function, optimizer):
    model.train(True)
    running_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = running_loss / len(train_loader)

    return model, avg_loss

#######################
# VALIDATION FUNCTION #
#######################

def validate_one_epoch(model, test_loader, loss_function):
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss = running_loss / len(test_loader)

    return avg_loss

###############
# TRAIN MODEL #
###############

learning_rate = 0.01
num_epochs = 200

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = LSTM(1, 4, 1).to(device)
loss_function = nn.L1Loss(reduction="mean")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model, avg_train_loss = train_one_epoch(model, train_loader, loss_function, optimizer)
    avg_val_loss = validate_one_epoch(model, test_loader, loss_function)
    print(f"Epoch: {epoch}, Train Loss:{avg_train_loss}, Val Loss: {avg_val_loss}")
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

plt.plot(train_losses)
plt.plot(val_losses)
plt.show()

#########################
# SAVING THE PARAMETERS #
#########################

torch.save(model.state_dict(), 'lstm.tar')



#### MODEL STARTS OVERFITTING AT 100 EPOCHS 
#### NEED TO APPLY REGULARIZATION OR OTHER METHODS TO AVOID OVERFITTING
#### OTHERWISE MODEL WORKING FINE

##############
# EVALUATION #
##############

with torch.no_grad():
    predicted = model(X_train.to(device)).to('cpu').numpy()

test_predictions = model(X_test.to(device)).detach().cpu().numpy().flatten()

dummies = np.zeros((X_test.shape[0], lookback+1))
dummies[:, 0] = test_predictions
dummies = scaler.inverse_transform(dummies)

test_predictions = dc(dummies[:, 0])
test_predictions[test_predictions >= 100] = 100
test_predictions = test_predictions[1:]

dummies = np.zeros((X_test.shape[0], lookback+1))
dummies[:, 0] = y_test.flatten()
dummies = scaler.inverse_transform(dummies)

new_y_test = dc(dummies[:, 0])
new_y_test = new_y_test[:-1]

plt.plot(new_y_test[-100:], label='Actual')
plt.plot(test_predictions[-100:], label='Predicted')
plt.xlabel('Day')
plt.ylabel('SQ')
plt.legend()
plt.show()