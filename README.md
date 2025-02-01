# Diabetes-Prediction

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
data=pd.read_csv("/content/diabetes.csv")
data

data.isnull().sum()
X=data.iloc[:, :-1].values #features
Y=data.iloc[:, -1].values  #labels

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

sh = 8
model = Sequential()
model.add(Dense(10,input_dim=sh,activation="relu"))
model.add(Dense(5,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

model.summary()

model.compile(optimizer ='Adam',loss="BinaryCrossentropy",metrics = ['accuracy'])
model.fit(X_train,y_train,epochs = 100)

graph = model.history.history

import matplotlib.pyplot as plt

plt.plot(graph['loss'])

```
