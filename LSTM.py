'''
An LSTM network is used to predict hourly load values. 
'''

import csv
load=open('F:/to do/load/kaggle/Load_history1.csv')
lread=csv.reader(load)
ldata=list(lread)

temp=open('F:/to do/load/kaggle/temperature_history1.csv')
tread=csv.reader(temp)
tdata=list(tread)

import numpy as np
import matplotlib.pyplot as plt

#Input data
l=np.asarray(ldata)
l1=l[1:65,50:]
l1=l1.flatten()
l1=np.array(l1).astype(float)

#l1=np.reshape(l1,(969,40))
l_test=l[66:98,50:]
l_test=l_test.flatten()
l_test=np.array(l_test).astype(float)

#plot_data=l[1:1633,50:]
#plot_data=plot_data.flatten()
#plot_data=np.array(plot_data).astype(float)

# Scaling the inputs
from sklearn.preprocessing import MinMaxScaler 
scaler=MinMaxScaler(feature_range=(0,1))
l1=scaler.fit_transform(l1)
l_test=scaler.fit_transform(l_test)
#plot_data=scaler.fit_transform(plot_data)

from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Flatten
from keras import optimizers,metrics
from keras.layers.recurrent import LSTM

def create_dataset(dataset, past):
	dataX, dataY = [], []
	for i in range(len(dataset)-past-1):
		a = dataset[i:(i+past)]
		dataX.append(a)
		dataY.append(dataset[i + past])
	return np.array(dataX), np.array(dataY)

past = 31
trainX, trainY = create_dataset(l1, past)
testX, testY = create_dataset(l_test, past)


trainX = np.reshape(trainX, (trainX.shape[0], 31, 1))
testX = np.reshape(testX, (testX.shape[0], 31, 1))

#Model is created and trained
model=Sequential()
model.add(LSTM(24,batch_input_shape=(32,31,1),return_sequences=True,stateful=True))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1,activation='linear'))

model.compile(loss='mse',optimizer='rmsprop')
model.fit(trainX,trainY,epochs=20,batch_size=32,shuffle=False)
   


testPredict = model.predict(testX)
print(model.evaluate(testX,testY))

testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
testY=np.transpose(testY)

# plot baseline and predictions
plt.plot(testY,color='blue')
plt.plot(testPredict,color='red')
plt.show()
