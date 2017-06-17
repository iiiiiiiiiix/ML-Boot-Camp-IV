import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

X = pd.read_csv('C:/Users/admin/Dropbox/docs/Data/ML Cup 4/x_train.csv', sep=';', header=None)
test = pd.read_csv('C:/Users/admin/Dropbox/docs/Data/ML Cup 4/x_test.csv', sep=';', header=None)
y = pd.read_csv('C:/Users/admin/Dropbox/docs/Data/ML Cup 4/y_train.csv', sep=';', header=None)
y = np.ravel(y)

X1 = X.as_matrix()

#отбор признаков
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

clf = RandomForestClassifier(random_state=16)
cv = StratifiedKFold(n_splits=7, random_state=16, shuffle=True)

sfs1 = SFS(clf, 
           k_features=46, 
           forward=True, 
           verbose=2,
           scoring='accuracy',
           cv=cv)  

start_time = time.time()

sfs1 = sfs1.fit(X1, y)

minutes = (time.time() - start_time) / 60
print("%.2f minutes" % minutes)

print(sfs1.subsets_)

X1 = X.iloc[:, [96, 131, 200, 138, 11, 76, 182, 156]]
test1 = test.iloc[:, [96, 131, 200, 138, 11, 76, 182, 156]]

rf = RandomForestClassifier(n_estimators=1000, random_state=16)
cv = StratifiedKFold(n_splits=7, random_state=16, shuffle=True)

print("cv:  ", cross_val_score(rf, X1, y, cv=cv, scoring='accuracy').mean() )
# cv: 0.6325

rf.fit(X1, y)
pred = rf.predict(test1)
np.savetxt('C:/Users/admin/Downloads/rf.csv', pred, delimiter=',')

