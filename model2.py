import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
df=pd.read_csv('Real_Combine.csv')
df.isnull().sum()
df=df.dropna()

Xx=df.iloc[:,:-1] ## independent features
yy=df.iloc[:,-1] 

from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(Xx,yy)
from sklearn.model_selection import train_test_split
Xx_train, Xx_test, yy_train, yy_test = train_test_split(Xx, yy, test_size=0.3, random_state=0)

import xgboost as xgb

regressor=xgb.XGBRegressor()
regressor.fit(Xx_train,yy_train)

regressor.score(Xx_train,yy_train)
from sklearn.model_selection import cross_val_score
score=cross_val_score(regressor,Xx,yy,cv=5)
#Evaluating The Model
prediction=regressor.predict(Xx_test)
import numpy as np
Xx_train=np.array(Xx_train)

from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
learning_rate = ['0.05','0.1', '0.2','0.3','0.5','0.6']
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
subsample=[0.7,0.6,0.8]
min_child_weight=[3,4,5,6,7]


random_grid = {'n_estimators': n_estimators,
               'learning_rate': learning_rate,
               'max_depth': max_depth,
               'subsample': subsample,
               'min_child_weight': min_child_weight}

regressor=xgb.XGBRegressor()
xg_random = RandomizedSearchCV(estimator = regressor, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
xg_random.fit(Xx_train,yy_train)

file = open('xgboost_random_model.pkl', 'wb')

pickle.dump(xg_random, file)