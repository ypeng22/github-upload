import numpy as np
import pandas as pd
from sklearn import preprocessing
import random 

random.seed(10)
###Data processing
X = np.load('x_sum.npy')
label = np.load('y_sum.npy')
CrByDay = pd.read_csv('CrbyDayInter.csv')
CrByDay = CrByDay[CrByDay.columns[1:8]]
CrByDay = pd.concat([pd.DataFrame(CrByDay.iloc[0]).transpose(), CrByDay])
CrByDay = np.array(CrByDay)
SD = pd.read_csv('labSD.csv')
SD.pop('aki')
SD = pd.concat([pd.DataFrame(SD.iloc[0]).transpose(), SD])
SD = np.array(SD)
for i in range(len(X)):
   X[i] = preprocessing.scale(X[i])
SD = preprocessing.scale(SD)

r = random.sample(range(len(X)), k = int(.8 * len(X)))
traindata = X[r]
trainSD = SD[r]
trainlabel = label[r]
trainday = CrByDay[r]
leftover = np.delete(np.array(range(len(X))), r)
testdata = X[leftover] 
testSD = SD[leftover]
testlabel = label[leftover]
testday = CrByDay[leftover]
#oversample imbalanced class
for i in range(len(traindata)):
    if trainlabel[i] == 1:
        for j in range(15):
            traindata = np.concatenate((traindata, [traindata[i]]))
            trainlabel = np.concatenate((trainlabel, [trainlabel[i]]))
            trainday = np.concatenate((trainday, [trainday[i]]))
            trainSD = np.concatenate((trainSD, [trainSD[i]]))

random.seed(10)    
#shuffle
indicies = np.arange(len(traindata))
random.shuffle(indicies)
traindata = traindata[indicies]
trainSD = trainSD[indicies]
trainlabel = trainlabel[indicies]
trainday = trainday[indicies]

np.save('trainx', traindata) #time series
np.save('trainlabel', trainlabel) #binary label
np.save('trainlabSD', trainSD) #static data
np.save('trainday', trainday) #target for 7 day Cr predictions 

np.save('testx', testdata)
np.save('testlabel', testlabel)
np.save('testlabSD', testSD)
np.save('testday', testday)


##Apend static data and time series data
trainxSD = np.zeros((len(traindata), 539, len(traindata[0][0]) + len(trainSD[0])))
testxSD = np.zeros((len(testdata), 539, len(traindata[0][0]) + len(trainSD[0])))
#append static to time series
for i in range(len(traindata)):
    for j in range(539):
        trainxSD[i][j] = np.concatenate([traindata[i][j], trainSD[i]])
        
for i in range(len(testdata)):
    for j in range(539):
        testxSD[i][j] = np.concatenate([testdata[i][j], testSD[i]])        
np.save('trainxSD', trainxSD)
np.save('testxSD', testxSD)


random.seed(11)
#split into 2 parts, for random forest and lstm
r = random.sample(range(len(trainSD)), k = int(.5 * len(trainSD)))
trainxSD1 = trainxSD[r]
trainSD1 = trainSD[r]
trainlabel1 = trainlabel[r]
leftover2 = np.delete(np.array(range(len(trainxSD))), r)
trainxSD2 = trainxSD[leftover2] 
trainSD2 = trainSD[leftover2]
trainlabel2 = trainlabel[leftover2]

np.save('trainxSD1', trainxSD1)
np.save('trainSD1', trainSD1)
np.save('trainlabel1', trainlabel1)
np.save('trainxSD2', trainxSD2)
np.save('trainSD2', trainSD2)
np.save('trainlabel2', trainlabel2)