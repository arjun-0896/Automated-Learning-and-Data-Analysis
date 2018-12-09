#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor                                                           
from sklearn import metrics

df = pd.read_csv("kc_house_data.csv")

nf = df
date_string = nf['date'].tolist()
date_int = [int(str[:4]) for str in date_string]    
df['year'] = date_int


xf = df
xf = xf.drop(['id','date','price','zipcode'],1)

yf = df
yf = yf.iloc[:,2] 


xs=(xf-xf.mean())/(xf.std())


rfreg = RandomForestRegressor(n_estimators=10)

X_train, X_test = train_test_split(xs, train_size=0.80, random_state=0)
Y_train, Y_test = train_test_split(yf, train_size=0.80, random_state=0)


rfreg.fit(X_train, Y_train)
y_pred = rfreg.predict(X_test)  

importance = rfreg.feature_importances_
score = rfreg.score(X_test, Y_test)
absolute_error = metrics.mean_absolute_error(Y_test, y_pred)
mse = metrics.mean_squared_error(Y_test, y_pred)
rmse = np.sqrt(mse)

print("\nThe importance of all attributes is given by ",importance)
print("\nNumber of estimators : 10")
print("\nThe score given by Random forest regressor is ", round( score,2))
print("\nMean Absolute Error is ", round(absolute_error,2))    
print("\nRoot Mean Squared Error is ", round(rmse,2))  


# In[6]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor                                                           
from sklearn import metrics



df = pd.read_csv("kc_house_data.csv")


nf = df
date_string = nf['date'].tolist()
date_int = [int(str[:4]) for str in date_string]    
df['year'] = date_int


xf = df
xf = xf.drop(['id','date','price','zipcode'],1)

yf = df
yf = yf.iloc[:,2] 

xs=(xf-xf.mean())/(xf.std())


rfreg = RandomForestRegressor(n_estimators=50)

X_train, X_test = train_test_split(xs, train_size=0.80, random_state=0)
Y_train, Y_test = train_test_split(yf, train_size=0.80, random_state=0)

rfreg.fit(X_train, Y_train)
y_pred = rfreg.predict(X_test)  

importance = rfreg.feature_importances_
score = rfreg.score(X_test, Y_test)
absolute_error = metrics.mean_absolute_error(Y_test, y_pred)
mse = metrics.mean_squared_error(Y_test, y_pred)
rmse = np.sqrt(mse)

print("\nThe importance of all attributes is given by ",importance)
print("\nNumber of estimators : 50")
print("\nThe score given by Random forest regressor is ", round( score,2))
print("\nMean Absolute Error is ", round(absolute_error,2))    
print("\nRoot Mean Squared Error is ", round(rmse,2))  


# In[5]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor                                                           
from sklearn import metrics



df = pd.read_csv("kc_house_data.csv")

nf = df
date_string = nf['date'].tolist()
date_int = [int(str[:4]) for str in date_string]    
df['year'] = date_int

xf = df
xf = xf.drop(['id','date','price','zipcode'],1)

yf = df
yf = yf.iloc[:,2] 

xs=(xf-xf.mean())/(xf.std())


rfreg = RandomForestRegressor(n_estimators=100)

X_train, X_test = train_test_split(xs, train_size=0.90, random_state=0)
Y_train, Y_test = train_test_split(yf, train_size=0.90, random_state=0)

rfreg.fit(X_train, Y_train)
y_pred = rfreg.predict(X_test)  

importance = rfreg.feature_importances_
score = rfreg.score(X_test, Y_test)
absolute_error = metrics.mean_absolute_error(Y_test, y_pred)
mse = metrics.mean_squared_error(Y_test, y_pred)
rmse = np.sqrt(mse)


print("\nThe importance of all attributes is given by ",importance)
print("\nNumber of estimators : 100")
print("\nThe score given by Random forest regressor is ", round( score,2))
print("\nMean Absolute Error is ", round(absolute_error,2))    
print("\nRoot Mean Squared Error is ", round(rmse,2))  


# In[3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor                                                           
from sklearn import metrics


df = pd.read_csv("kc_house_data.csv")

nf = df
date_string = nf['date'].tolist()
date_int = [int(str[:4]) for str in date_string]    
df['year'] = date_int

xf = df
xf = xf.drop(['id','date','price','zipcode'],1)

yf = df
yf = yf.iloc[:,2] 

xs=(xf-xf.mean())/(xf.std())

rfreg = RandomForestRegressor(n_estimators=200)

X_train, X_test = train_test_split(xs, train_size=0.90, random_state=0)
Y_train, Y_test = train_test_split(yf, train_size=0.90, random_state=0)

rfreg.fit(X_train, Y_train)
y_pred = rfreg.predict(X_test)  

importance = rfreg.feature_importances_
score = rfreg.score(X_test, Y_test)
absolute_error = metrics.mean_absolute_error(Y_test, y_pred)
mse = metrics.mean_squared_error(Y_test, y_pred)
rmse = np.sqrt(mse)


print("\nThe importance of all attributes is given by ",importance)
print("\nNumber of estimators : 200")
print("\nThe score given by Random forest regressor is ", round( score,2))
print("\nMean Absolute Error is ", round(absolute_error,2))    
print("\nRoot Mean Squared Error is ", round(rmse,2))  


# In[ ]:




