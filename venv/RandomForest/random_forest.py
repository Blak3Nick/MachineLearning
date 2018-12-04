#Polynomial Regression Template
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Feature Scaling
"""
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)
"""

#encoding the categories
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder_X = LabelEncoder()
# X[:, 3]= labelencoder_X.fit_transform(X[:, 3])
# onehotencoder = OneHotEncoder(categorical_features=[3])
# X = onehotencoder.fit_transform(X).toarray()
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=1000, random_state=0)
regressor.fit(X, y)
y_pred = regressor.predict([[6.5]])


#Visualize the polynomial results
#for higher resolution include the next two lines
X_grid = np.arange(min(X), max(X), 0.001)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='green')
plt.title('Random Forest')
plt.xlabel('Position level with .1 steps')
plt.ylabel('Salary')
plt.show()