import pandas as pd
import numpy as np

df = pd.read_csv('lab.csv') #df = pd.read_csv('labfixed2.csv', index_col=0)
l = len(df)
meas = dict()

time = df['dt']
name = df['name']
cases = df['caseid']
prev = -1

for i in range(l):
    if time.iloc[i] > 0:
        try:
            meas[name.iloc[i]] += 1
        except:
            meas[name.iloc[i]] = 0
for w in sorted(meas, key=meas.get, reverse=True):
    print(w, meas[w])

     
unique = dict()       
temp = set()    
for i in range(l):
    if time.iloc[i] > 0:
        if cases.iloc[i] == prev:
           temp.add(name.iloc[i])
        else:
            #print(prev)
            for k in temp:
                try:
                    unique[k] += 1
                except:
                    unique[k] = 0
            temp = set()
            prev = cases.iloc[i]
            
    
for w in sorted(unique, key=unique.get, reverse=True):
    print(w, unique[w])   