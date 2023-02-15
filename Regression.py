import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error
import statsmodels.api as sm

#are there missing values

#df.isna().sum - will determine if there are any missing values
# df = df.fillna(0) - will fill missing values with 0

df = pd.read_csv('data.csv')
df = df.drop(columns="Unamed")
df.isna().sum()
print(df)
df.isna().sum()
df = df.fillna(0)


#Pairwise correlations
le = LabelEncoder()

le.fit(df['Purchase'])
df['Purchase'] = le.transform(df['Purchase'])

le.fit(df['Duration'])
df['Duration'] = le.transform(df['Duration'])

le.fit(df['Gender'])
df['Gender'] = le.transform(df['Gender'])

le.fit(df['ASR_Error'])
df['ASR_Error'] = le.transform(df['ASR_Error'])

le.fit(df['Intent_Error'])
df['Intent_Error'] = le.transform(df['Intent_Error'])

print(df.corr(method='pearson')['SUS'].sort_values())

#OLS Regression
y = df['SUS'] # dependent variable
x = df.drop(columns='SUS') # predictor variables

x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
print(model.summary())


#Linear regression model
x = df.drop(columns='SUS')
y = df['SUS']

x_train, x_test, y_train, y_test = train_test_split(x, y)

lr = LinearRegression().fit(x_train,y_train)

y_train_pred = lr.predict(x_train)
y_test_pred = lr.predict(x_test)

print("The R square score of linear regression model is: ", lr.score(x_test,y_test))