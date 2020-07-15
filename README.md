# Salary-prediction---Linear-regression-with-Categorical-variable
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

df=df = pd.read_excel('Multiple_variable.xlsx',sheet_name='Sheet2')
df.head()

df.shape
df.describe()

sns.pairplot(df,x_vars=['Age','YearsExperience'],y_vars=['Salary'],hue='Gender')

X = df[['Age', 'YearsExperience', 'Gender', 'Classification', 'Job']]
X = pd.get_dummies(data=X, drop_first=True)
X.head()
Y = df['Salary']
Y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=101)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
# print the intercept
print(model.intercept_)
coeff_parameter = pd.DataFrame(model.coef_,X.columns,columns=['Coefficient'])
coeff_parameter
predictions = model.predict(X_test)
predictions
sns.distplot(predictions,bins=3)
sns.regplot(y_test,predictions)
from sklearn import metrics
import numpy as np
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
import statsmodels.api as sm
X_train_Sm= sm.add_constant(X_train)
X_train_Sm= sm.add_constant(X_train)

ls=sm.OLS(y_train,X_train_Sm).fit()
ls.params

from statsmodels.formula.api import ols
fit = ols('Salary ~ C(Gender) + C(Job) + Age', data=df).fit() 
print(fit.summary())
