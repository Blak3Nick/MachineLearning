import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the data
df = dataset = pd.read_csv('Data.csv')
#print(df)

X = dataset.iloc[:, :-1].values
#print(X)

#missing data solution
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)
