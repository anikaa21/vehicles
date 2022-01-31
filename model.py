import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import VotingRegressor
        
df = pd.read_csv("FuelConsumption.csv", encoding= 'unicode_escape')
df.replace([np.inf, -np.inf], np.NaN, inplace=True)
df.fillna(999, inplace=True)
df.head()
cdf = df[['Engine_Size','Cylinders','Fuel_Consumption_city ','Fuel_consumption_Hwy','Fuel_consumption_Comb','CO2_Emissions','Smog ']]
cdf.head(9)
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

gb = GradientBoostingRegressor(n_estimators=200, 
                        max_depth=5, learning_rate=0.1, random_state=101)
x = np.asanyarray(train[['Engine_Size','Cylinders','Fuel_Consumption_city ','Fuel_consumption_Hwy']])
y = np.asanyarray(train[['CO2_Emissions']])

gb.fit(x, y)
linear_model = LinearRegression()
linear_model.fit(x,y)
linearSVR_model=LinearSVR()
linearSVR_model.fit(x, y)

ensemble_model = VotingRegressor([
            ("linear", linear_model), 
            ("linearSVRmodel",linearSVR_model ),
            ("GB",gb)
        ])
    
ensemble_model.fit(x, y)

pickle.dump(ensemble_model,open('ensemble_model.pkl','wb'))
model=pickle.load(open('ensemble_model.pkl','rb'))




