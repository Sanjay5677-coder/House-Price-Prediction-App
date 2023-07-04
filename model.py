import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("kc_house_data.csv")
df.drop(['id','date'],axis=1,inplace=True)
df.rename(columns={'sqft_living':'Living Area'},inplace=True)
df.drop(['yr_built'],axis=1,inplace=True)
df.fillna(0,inplace=True)
df.drop(['sqft_lot','condition','grade','sqft_above','zipcode','lat','long'],axis=1,inplace=True)
x  = df.iloc[:,1:]
y = df['price']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=15)
from sklearn.ensemble import RandomForestRegressor
newModel = RandomForestRegressor()
newModel.fit(x_train,y_train)
y_pred = newModel.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
import pickle
model = pickle.dump(newModel,open('NewModel.pkl','wb'))
print((y_pred))
