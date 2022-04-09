
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv("FuelConsumption.csv", encoding= 'unicode_escape')
df.replace([np.inf, -np.inf], np.NaN, inplace=True)
df.fillna(999, inplace=True)
df.head()
cdf = df[['Engine_Size','Cylinders','Fuel_Consumption_city','Fuel_consumption_Hwy','Fuel_consumption_Comb','CO2_Emissions','Smog ']]
cdf.head(9)
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
x = np.asanyarray(train[['Engine_Size','Cylinders','Fuel_Consumption_city','Fuel_consumption_Hwy']])
y = np.asanyarray(train[['CO2_Emissions']])
RandomForest_Regressor = RandomForestRegressor()
RandomForest_Regressor.fit(x,y)
DecisionTreeRegressor_Model=DecisionTreeRegressor()
DecisionTreeRegressor_Model.fit(x, y)
ensemble_model = VotingRegressor([
        ("RandomForestregressor", RandomForest_Regressor), 
        ("DecisionTreeRegressor",DecisionTreeRegressor_Model),
        ])

ensemble_model.fit(x, y)

pickle.dump(ensemble_model,open('ensemble_modelCO2.pkl','wb'))
model=pickle.load(open('ensemble_modelCO2.pkl','rb'))

