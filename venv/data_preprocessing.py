import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = dataset = pd.read_csv('Data.csv')
print(df)

X = dataset.iloc[:, :-1].values
print(X)
