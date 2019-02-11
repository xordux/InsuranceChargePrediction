# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('insurance.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X[:,1] = LabelEncoder().fit_transform(X[:,1])
X[:,4] = LabelEncoder().fit_transform(X[:,4])
X[:,5] = LabelEncoder().fit_transform(X[:,5])

OHE_X = OneHotEncoder(categorical_features=[5])
X = OHE_X.fit_transform(X).toarray()

#Avoding Dummy Variable Trap (Although sklearn takes care of this by itself, the below line is just to be sure)
X=X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((1338,1)).astype(int), values = X, axis = 1)

X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 4, 6, 7, 8]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 2, 3, 4, 6, 7, 8]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 2, 4, 6, 7, 8]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 4, 6, 7, 8]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


#Regression using column 0, 3 and 5 only
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train[:, [3, 5, 6, 7]], y_train)
y_predt = regressor.predict(X_test[:, [3, 5, 6, 7]])

from sklearn.metrics import mean_squared_error, r2_score
print("Mean Squared error is: ", mean_squared_error(y_true=y_test, y_pred=y_predt))

print("R2 Score is:", r2_score(y_true=y_test, y_pred=y_predt))

plt.scatter(range(len(y_test)), y_test, color="red")
plt.scatter(range(len(y_predt)), y_predt, color="black")
plt.show()