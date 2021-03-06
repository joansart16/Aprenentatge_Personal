import pandas as pd

# Llibreries que necessitarem
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

random_value = 33
# Carrega de dades i preparaci√≥ de les dades emprant Pandas
data = pd.read_csv("data/day.csv")
datos = pd.DataFrame(data.iloc[:, 4:13])  # Seleccionam totes les files i les columnes per index
valors = pd.DataFrame(data.iloc[:, -1])  # Seleccionam totes les files i la columna objectiu

X = datos.to_numpy()
y = valors.to_numpy().ravel()
features_names = datos.columns

#Separaci√≥ de dades: entrenament i test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)
estimator = RandomForestRegressor()

param_grid = {"estimator": [10,20,40,80,100,120,140,160,200], "min_samples_split":[2,4,8,16], "min_samples_leaf":[1,2,4,8,16,32], "max_features": ["auto", "sqrt", "log2"]}

estimator.fit(X,y)
print(estimator.get_params())
grid = GridSearchCV(estimator, features_names, cv=None, verbose=0)
grid.decision_function(X,y)
best_est = grid.best_estimator_
best_est.fit(X,y)

print(best_est.score(X,y))

#regression forest
