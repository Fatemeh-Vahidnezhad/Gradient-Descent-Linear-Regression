import pandas as pd
import numpy as np 
from LinearRegression_dataframe import *
from sklearn.preprocessing import StandardScaler

# data sample:
path = 'advertising.csv'
df = pd.read_csv(path)
scaler = StandardScaler()
df = scaler.fit_transform(df)
target = df[:,-1]
column_to_select = [0,1,2]
data = df[:, column_to_select]
lr = LinearRegression(data, target, learning_rate=0.00001)
lr.train(10000,500)
y_pred = lr.prediction()
print(lr.MSE())