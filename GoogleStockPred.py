import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

training_set = pd.read_csv('Google_Stock_Price_Train.csv')
test_set = pd.read_csv('Google_Stock_Price_Test.csv')

training_set.head()
training_set.info()

training_set = training_set.iloc[:,1:2].values
real_stock_price = test_set.iloc[:,1:2].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

X_train = training_set[0:1257]
Y_train = training_set[1:1258]

X_train = np.reshape(X_train, (1257,1,1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

regressor = Sequential()

regressor.add(LSTM(units=4, activation = 'sigmoid', input_shape=(None,1)))

regressor.add(Dense(units=1))
regressor.compile(optimizer= 'adam', loss = 'mean_squared_error')
regressor.fit(X_train,Y_train,batch_size=32,epochs=200)

print(len(real_stock_price))

inputs = real_stock_price
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (20,1,1))
predicted = regressor.predict(inputs)
predicted = sc.inverse_transform(predicted)

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock')
plt.plot(predicted, color = 'blue', label = 'Predicted Google Stock')
plt.xlabel('Time')
plt.ylabel('Stock')
plt.legend()
plt.show()
