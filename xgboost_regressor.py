import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
df=pd.read_csv('Real_Combine.csv')
df.head()

df.isnull().sum()

sns.heatmap(df.isnull(),yticklabels=False,cmap="viridis")

df=df.dropna()

X=df.iloc[:,:-1] ## independent features
y=df.iloc[:,-1]

sns.pairplot(df)

df.corr()

plt.figure(figsize=(14,12))
sns.heatmap(df.corr(),annot=True,cmap="RdYlGn")

from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(X,y)

X.head()

model.feature_importances_

X.head(2) #eg T=0.14 Tm= 0.08

fea_imp=pd.Series(model.feature_importances_,index=X.columns)
fea_imp

fea_imp=pd.Series(model.feature_importances_,index=X.columns)
fea_imp

fea_imp.nlargest(6).plot(kind='barh')



"""# Performing Linear Regression on the model"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

regressor.coef_

regressor.intercept_

regressor.score(X_train,y_train)

regressor.score(X_test,y_test)

from sklearn.model_selection import cross_val_score
score=cross_val_score(regressor,X,y,cv=5)

score

score.mean()



pd.DataFrame(regressor.coef_,columns=['Coeff'])

pred=regressor.predict(X_test)

sns.distplot(y_test-pred)



from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, pred)) #Mean absolute error
print('MSE:', metrics.mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))

"""# XGBoost Regressor"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

import xgboost as xgb
xgb.__version__

regressor=xgb.XGBRegressor()
regressor.fit(X_train,y_train)

regressor.score(X_train,y_train)

regressor.score(X_test,y_test)

from sklearn.model_selection import cross_val_score
score=cross_val_score(regressor,X,y,cv=5)

score

score.mean()

#Evaluating The Model
prediction=regressor.predict(X_test)

sns.distplot(y_test-prediction)



#Hyper Parameter Tuning

import numpy as np
X_train=np.array(X_train)

X_train
#X_test=X_test.values
#y_train=y_train.values
#y_test=y_test.values

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

xg_random.fit(X_train,y_train)

xg_random.best_params_

xg_random.best_score_

predictions=xg_random.predict(X_test.values)

from sklearn.metrics import r2_score
r2_score(y_test,predictions)

sns.distplot(y_test-predictions)

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

y.head()


a=np.array([[30,35,27,1000,40,.0,12.0,15.0]])

a=np.array([[24,35,20,1000,40,1.5,7.0,16.0]])
xg_random.predict(a)

#import pickle

# open a file, where you ant to store the data
file = open('xgboost_random_model.pkl', 'wb')

# dump information to that file
pickle.dump(xg_random, file)







