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
from sklearn import preprocessing

house_data=pd.read_csv('Housing.csv')


class model():

    def train(input):
     

        house_data=pd.read_csv('Housing.csv')
        house_data['Price_per_sqft'] = house_data['price']/house_data['area']

        label_encoder = preprocessing.LabelEncoder()

        for col in house_data.columns:
            if house_data[col].dtype == 'object': 
                house_data[col] = label_encoder.fit_transform(house_data[col])


     
        X=house_data.drop(columns=['price'],axis=1)
        Y=house_data['price']
        X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=10)
        lr_clf = LinearRegression()
        lr_clf.fit(X_train,Y_train)
        LR_prediction=lr_clf.predict(X_test)
        DT = DecisionTreeRegressor()
        DT.fit(X_train,Y_train)
        DT_prediction=DT.predict(X_test)
        RF=RandomForestRegressor()
        RF.fit(X_train,Y_train)
        RF_prediction=RF.predict(X_test)
        RF.score(X_test,Y_test)
        pd.DataFrame(data={'Actual':Y_test,'Predicted':RF_prediction}).head()

        Input_data = np.array(input, dtype=float)  # Convert input to numpy array and ensure dtype is float
        reshaped_input = Input_data.reshape(1, -1)  # Reshape the input
        Predictive_system = RF.predict(reshaped_input)
        return(Predictive_system)





