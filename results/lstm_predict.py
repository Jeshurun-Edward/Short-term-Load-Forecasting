import csv
load=open('F:/projects/load/kaggle/Load_history1.csv')
lread=csv.reader(load)
ldata=list(lread)

temp=open('F:/projects/load/kaggle/temperature_history1.csv')
tread=csv.reader(temp)
tdata=list(tread)

import numpy as np
import matplotlib.pyplot as plt


l=np.asarray(ldata)
l1=l[1:1601,50:]
l1=l1.flatten()
l1=np.array(l1).astype(float)

#l1=np.reshape(l1,(969,40))
l_test=l[1501:1533,50:]
l_test=l_test.flatten()
l_test=np.array(l_test).astype(float)

plot_data=l[1:1533,50:]
plot_data=plot_data.flatten()
plot_data=np.array(plot_data).astype(float)

#scaling
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
l1=scaler.fit_transform(l1)
l_test=scaler.fit_transform(l_test)
plot_data=scaler.fit_transform(plot_data)

from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Flatten
from keras import optimizers,metrics
from keras.layers.recurrent import LSTM

def create_dataset(dataset, look_back):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return np.array(dataX), np.array(dataY)

look_back = 31
trainX, trainY = create_dataset(l1, look_back)
testX, testY = create_dataset(l_test, look_back)


trainX = np.reshape(trainX, (trainX.shape[0], 31, 1))
testX = np.reshape(testX, (testX.shape[0], 31, 1))

#Trained Model is Loaded
from sklearn.metrics import mean_squared_error
import math
from keras import metrics
from keras.models import load_model
model=load_model('F:/projects/load/anaconda/lstm.h5')

score=model.evaluate(testX,testY)
print(score)


#True values from scaled values
pred=model.predict(testX)
pred = scaler.inverse_transform(pred)
testY = scaler.inverse_transform([testY])
testY=np.transpose(testY)
rms=math.sqrt(mean_squared_error(testY[1:169,:],pred[1:169,:]))
print(rms)
 
# plot baseline and predictions
plt.plot(testY[1:169,:],color='blue')
plt.plot(pred[1:169,:],color='red')
plt.show()

