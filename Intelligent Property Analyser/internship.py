# -*- coding: utf-8 -*-
"""TCS_INTERNSHIP.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aJhoy4Z9n23U5yGF7yOF3iVj4xP69XIY

### Importing Liabraries
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from math import sqrt

house_data=pd.read_csv('Housing.csv')

house_data

"""### DATA PREPROCESSING"""

house_data.head()

house_data.tail()

house_data.shape

house_data.columns

house_data.info()

house_data['bedrooms'].value_counts()

house_data['bathrooms'].value_counts()

house_data['furnishingstatus'].value_counts()

house_data.duplicated().sum()

"""### Checking for null values"""

house_data.isnull().sum()

house_data['Price_per_sqft'] = house_data['price']/house_data['area']

"""#### making price per square fit for houses by dividing actual price with area"""

house_data['Price_per_sqft']

"""### DATA VISUALIZATION

#### Univariate Analysis
"""

plt.figure(figsize=(10,8))
labels =house_data['bedrooms'].value_counts(sort = True).index
sizes = house_data['bedrooms'].value_counts(sort = True)
explode = (0.0, 0.0, 0.0, 0.3, 0.3, 0.4)
plt.pie(sizes, labels=labels,autopct='%1.1f%%', shadow=True, startangle=270,explode=explode)
plt.title('Total % of Bedrooms',size = 20)
plt.show()

house_data['furnishingstatus'].value_counts().plot(kind='bar')
plt.show()

"""#### Bivariate Analysis"""

plt.figure(figsize=(10,8))
sns.scatterplot(x='area',y='price',data=house_data,hue='bedrooms',s=90)
plt.xlabel("Total Sqfeet Area",size=15,c="r")
plt.ylabel("Avg Price (Lakh)",size=15,c="r")
plt.show()

df_1=house_data[house_data['bedrooms']==1]
df_2=house_data[house_data['bedrooms']==2]
df_3=house_data[house_data['bedrooms']==3]
df_4=house_data[house_data['bedrooms']==4]
df_5=house_data[house_data['bedrooms']==5]
df_6=house_data[house_data['bedrooms']==6]
plt.figure(figsize=(15,15))
plt.subplot(4,2,1)
sns.boxplot(data=df_1,x=df_1['price'])
plt.subplot(4,2,2)
sns.boxplot(data=df_2,x=df_2['price'])
plt.subplot(4,2,3)
sns.boxplot(data=df_3,x=df_3['price'])
plt.subplot(4,2,4)
sns.boxplot(data=df_4,x=df_4['price'])
plt.subplot(4,2,5)
sns.boxplot(data=df_5,x=df_5['price'])
plt.subplot(4,2,6)
sns.boxplot(data=df_6,x=df_6['price'])
plt.show()

"""#### Multivariate Analysis"""

label_encoder = preprocessing.LabelEncoder()

for col in house_data.columns:
    if house_data[col].dtype == 'object':
        house_data[col] = label_encoder.fit_transform(house_data[col])

house_corr=house_data.corr()
plt.figure(figsize=(10,10))
sns.heatmap(house_corr,square=True,cbar=True,annot=True,cmap='Blues')
plt.show()

house_data

"""### TRAIN_TEST_SPLIT"""

X=house_data.drop(columns=['price'],axis=1)
Y=house_data['price']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=10)

print(X.shape,X_train.shape,X_test.shape)

print(Y.shape,Y_train.shape,Y_test.shape)

"""### Model Building

##### LinearRegression
"""

lr_clf = LinearRegression()
lr_clf.fit(X_train,Y_train)

LR_prediction=lr_clf.predict(X_test)
print('R2 Score :',r2_score(LR_prediction,Y_test)*100)
print('MAE:',mean_absolute_error(LR_prediction,Y_test))
print('MSE:',mean_squared_error(LR_prediction,Y_test))
rmse = sqrt(mean_squared_error(LR_prediction,Y_test))
print('RMSE:',rmse)

"""##### DecisionTreeRegressor"""

DT = DecisionTreeRegressor()
DT.fit(X_train,Y_train)


DT_prediction=DT.predict(X_test)
print('R2 Score :',r2_score(DT_prediction,Y_test)*100)
print('MAE:',mean_absolute_error(DT_prediction,Y_test))
print('MSE:',mean_squared_error(DT_prediction,Y_test))
rmse = sqrt(mean_squared_error(DT_prediction,Y_test))
print('RMSE:',rmse)

"""##### RandomForestRegressor"""

RF=RandomForestRegressor()
RF.fit(X_train,Y_train)

RF_prediction=RF.predict(X_test)
print('R2 Score :',r2_score(RF_prediction,Y_test)*100)
print('MAE:',mean_absolute_error(RF_prediction,Y_test))
print('MSE:',mean_squared_error(RF_prediction,Y_test))
rmse = sqrt(mean_squared_error(RF_prediction,Y_test))
print('RMSE:',rmse)

RF.score(X_test,Y_test)

pd.DataFrame(data={'Actual':Y_test,'Predicted':RF_prediction}).head()

house_data.head(5)

"""### Building a predictive System"""

Input_data=np.array([[17420,4,2,3,1,0,0,0,1,2,1,0,1792.452830]])
Predictive_system=RF.predict(Input_data)
print(Predictive_system)