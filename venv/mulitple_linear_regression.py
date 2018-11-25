# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values



# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#encoding the categories
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3]= labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy variable trap-remove one of the dummy variables
"""most libraries do this automatically"""
X = X[:, 1:]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting the Multiple Linear Regression t othe Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test set results
Y_pred = regressor.predict(X_test)
#Building the optimal model using backward elimination
"""
Step 1 -Select a significance level to stay: i.e. SL = .05
Step 2 -Fit the model with all possible predictors
Step 3 -Consider the rpedictor with the highest P-value: If P > SL go to Step 4, else finished
Step 4 -Remove the predictor
Step 5 -Fit model without this variable, re-run from Step 3
"""
import statsmodels.formula.api as sm
#adding a 50 by 1 matrix of ones to the begining of X, this signifies Xsub 0
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
#use only the optimal variables
X_opt = X[:, [0, 1, 2, 3, 4, 5]]