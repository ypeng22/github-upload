import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from numpy import random
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import sys

df = pd.read_csv("labSD.csv")
labels = df.pop("aki")
df.pop('caseid')
df.pop('dis')
df.pop('death_inhosp')
#df.pop('optype')

c1 = df['age']
c2 = df['sex']
c3 = df['height']
c4 = df['weight']
c5 = df['age']
c6 = df['optype']
#c7 = df['anedur']
df = pd.DataFrame([c1, c2, c3, c4, c5, c6]).transpose()


cols = df.columns
train, test, train_labels, test_labels = train_test_split(df, labels, train_size=.4)
features = list(train.columns)
random.seed(49) #.5596638655462184
model = RandomForestClassifier(criterion='entropy', n_estimators=500, bootstrap=True, max_features='sqrt', class_weight = "balanced_subsample", n_jobs=-1, max_depth = 15)
model.fit(train, train_labels)
imp = model.feature_importances_;
indices = np.argsort(imp)[::-1]

for f in range(df.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, cols[indices[f]], imp[indices[f]]))
predict = model.predict_proba(test)[:,1]
print("AUC score: ", roc_auc_score(test_labels, predict))



scores = cross_validate(model, df, labels, scoring = 'roc_auc', n_jobs = -1, return_estimator =True);
print("HERE1: " ,scores['test_score']);
