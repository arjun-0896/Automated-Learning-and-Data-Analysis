#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

df = pd.read_csv("kc_house_data.csv")

# Plotting correlation heatmap to get the factors highly correlated with price
correlation = df.corr()
fig, ax = plt.subplots(figsize=(20, 10))
cm = sns.diverging_palette(5, 250, as_cmap=True)
sns.heatmap(correlation, cmap=cm, annot=True, fmt=".2f")    
plt.show()
plt.show()


# In[3]:


# Linear regression using sqft_living as X

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn import metrics

df = pd.read_csv("kc_house_data.csv")

rf = df.drop(['id','date','price'],1)

#  From the heat map we identified that sqft_living and grade have more correlation to price
# So we did two models 
# -> Using all attributes against price 
# -> Using sqft_living and grade attributes against price 


column_selected = ['sqft_living']
predicted_array = []

train, test = train_test_split(df, train_size =0.90, random_state = 3)

lm = linear_model.LinearRegression()

X_train = np.array(train[column_selected]).reshape(-1,1)
X_test = np.array(test[column_selected]).reshape(-1, 1)

Y_train = np.array(train['price']).reshape(-1, 1)
Y_test = np.array(test['price']).reshape(-1, 1)

lm.fit(train[column_selected], train['price'])

prediction = lm.predict(test[column_selected])

print("Model using {} as X" .format(column_selected))
mse = metrics.mean_squared_error(Y_test, prediction)
error = np.sqrt(mse)
intercept = lm.intercept_
accuracy = lm.score(test[column_selected], test['price'])
                       
print("\nThe root mean squared error is ", np.round(error, 2))
print("\nThe coefficient array is ", np.round(lm.coef_,2))
print("\nThe intercept is ", np.round(intercept,2))
print("\nThe accuracy is given by is ", round(accuracy, 2))

fig, ax = plt.subplots(figsize= (15, 10))
plt.scatter(X_test, Y_test, color= 'blue', label = 'Scattered Dataset')
plt.plot(X_test, lm.predict(X_test), color='black', label= 'Predicted Regression line')
plt.xlabel('Square ft Living')
plt.ylabel('Price of the house')
plt.legend()






# In[5]:


# Linear regression using sqft_living as X

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns


df = pd.read_csv("kc_house_data.csv")

rf = df.drop(['id','date','price'],1)


column_selected = ['grade']
predicted_array = []

train, test = train_test_split(df, train_size =0.90, random_state = 3)

lm = linear_model.LinearRegression()

X_train = np.array(train[column_selected]).reshape(-1,1)
X_test = np.array(test[column_selected]).reshape(-1, 1)

Y_train = np.array(train['price']).reshape(-1, 1)
Y_test = np.array(test['price']).reshape(-1, 1)

lm.fit(train[column_selected], train['price'])

prediction = lm.predict(test[column_selected])

print("Model using {} as X" .format(column_selected))
mse = metrics.mean_squared_error(Y_test, prediction)
error = np.sqrt(mse)
intercept = lm.intercept_
accuracy = lm.score(test[column_selected], test['price'])
                       
print("\nThe root mean squared error is ", np.round(error, 2))
print("\nThe coefficient array is ", np.round(lm.coef_,2))
print("\nThe intercept is ", np.round(intercept,2))
print("\nThe accuracy is given by is ", round(accuracy, 2))


fig, ax = plt.subplots(figsize= (15, 10))
plt.scatter(X_test, Y_test, color= 'blue', label = 'Scattered Dataset')
plt.plot(X_test, lm.predict(X_test), color='black', label= 'Predicted Regression line')
plt.xlabel('Grade')
plt.ylabel('Price of the house')
plt.legend()
plt.show()






# In[ ]:





# In[6]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

df = pd.read_csv("kc_house_data.csv")

nf = df
date_string = nf['date'].tolist()
date_int = [int(str[:4]) for str in date_string]    
df['year'] = date_int

rf = df.drop(['id','date','price', 'zipcode'],1)

#  From the heat map we identified that sqft_living and grade have more correlation to price
# So we did two models 
# -> Using all attributes against price 
# -> Using sqft_living and grade attributes against price 

column_selected = list(rf.columns.values)

predicted_array = []

train, test = train_test_split(df, train_size =0.90, random_state = 3)

lm = linear_model.LinearRegression()

X_train = np.array(train[column_selected]).reshape(-1,1)
X_test = np.array(test[column_selected]).reshape(-1, 1)

Y_train = np.array(train['price']).reshape(-1, 1)
Y_test = np.array(test['price']).reshape(-1, 1)

lm.fit(train[column_selected], train['price'])

prediction = lm.predict(test[column_selected])

print("Model using {} as X" .format(column_selected))
mse = metrics.mean_squared_error(Y_test, prediction)
error = np.sqrt(mse)
intercept = lm.intercept_
accuracy = lm.score(test[column_selected], test['price'])
                       
print("\nThe root mean squared error is ", np.round(error, 2))
print("\nThe coefficient array is ", np.round(lm.coef_,2))
print("\nThe intercept is ", np.round(intercept,2))
print("\nThe accuracy is given by is ", round(accuracy, 2))


# In[ ]:




