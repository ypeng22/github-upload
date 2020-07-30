# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 16:02:48 2020

@author: Yu-Chung Peng
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 16:36:10 2020

@author: Yu-Chung Peng
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from numpy import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold

df = pd.read_csv("labSD.csv")
labels = df.pop("aki")
df.pop('caseid')
random.seed(60) 

model = RandomForestClassifier(n_jobs=4)
grid = {"n_estimators": [100, 300, 500], "criterion": ['gini', 'entropy'], "min_samples_split" : [2, 5, 10], "max_features" : ['sqrt', 'log2'], "max_depth": [3, 6, 9, 12, 15], "class_weight": ['balanced', 'balanced_subsample']} 
inner_cv = KFold(n_splits=3, shuffle=True)
clf = GridSearchCV(estimator = model, param_grid = grid, n_jobs = 4, cv = inner_cv, scoring='roc_auc')
clf.fit(df, labels)
bestmodel = clf.best_estimator_
print(clf.best_score_)
print(bestmodel.get_params())
