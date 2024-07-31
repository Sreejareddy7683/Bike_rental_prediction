# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/bipulshahi/Dataset/main/Daily%20Bike%20Sharing.csv')
print(df.head())

# Data preprocessing
df1 = df.drop('instant', axis='columns')
print(df1.head())

# Checking for missing values
print(df1.isna().sum())

# Displaying data types
print(df.dtypes)

# Displaying unique values in each column
print(df1.nunique())

# Converting date column to datetime
df1['dteday'] = pd.to_datetime(df1['dteday'], format='%Y-%m-%d')
day = df1['dteday'].dt.day
df1.insert(1, "Day", day)
print(df1.head())

# Dropping unnecessary columns
df2 = df1.drop(['dteday', 'yr'], axis='columns')
print(df2.head())

# Displaying correlation matrix
print(df2.corr())

# Plotting heatmap of correlation matrix
plt.figure(figsize=(10, 10))
sns.heatmap(df2.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

# Dropping 'casual' and 'registered' columns
df3 = df2.drop(['casual', 'registered'], axis='columns')
print(df3.head())

# Plotting histogram of the target variable
df3['cnt'].hist()

# Splitting data into features and target variable
X = df3.drop('cnt', axis='columns')
y = df3['cnt']

# Split the data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data scaling
scaler = MinMaxScaler()
X_trainScaled = scaler.fit_transform(X_train)
X_testScaled = scaler.transform(X_test)

# Linear Regression model
model1 = LinearRegression()
model1.fit(X_trainScaled, y_train)

# Making predictions
y_trainpred = model1.predict(X_trainScaled)
y_Testpred = model1.predict(X_testScaled)

# Evaluation - mean_absolute_error , r2_score
print('Linear Regression:')
print('Mean Absolute Error:', mean_absolute_error(y_train, y_trainpred))
print('Mean Absolute Error:', mean_absolute_error(y_test, y_Testpred))
print('R2 Score:', r2_score(y_train, y_trainpred))
print('R2 Score:', r2_score(y_test, y_Testpred))

# Non-linear 2 degree transformation
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_trainPoly = poly.fit_transform(X_trainScaled)
X_testPoly = poly.transform(X_testScaled)

# Polynomial Regression model
model2 = LinearRegression()
model2.fit(X_trainPoly, y_train)

# Making predictions
y_trainpred = model2.predict(X_trainPoly)
y_Testpred = model2.predict(X_testPoly)

# Evaluation
print('Polynomial Regression (degree 2):')
print('Mean Absolute Error:', mean_absolute_error(y_train, y_trainpred))
print('Mean Absolute Error:', mean_absolute_error(y_test, y_Testpred))
print('R2 Score:', r2_score(y_train, y_trainpred))
print('R2 Score:', r2_score(y_test, y_Testpred))

# Decision Tree Regressor model
model3 = DecisionTreeRegressor(max_depth=4)
model3.fit(X_trainScaled, y_train)

# Making predictions
y_trainpred = model3.predict(X_trainScaled)
y_Testpred = model3.predict(X_testScaled)

# Evaluation
print('Decision Tree Regressor:')
print('Mean Absolute Error:', mean_absolute_error(y_train, y_trainpred))
print('Mean Absolute Error:', mean_absolute_error(y_test, y_Testpred))
print('R2 Score:', r2_score(y_train, y_trainpred))
print('R2 Score:', r2_score(y_test, y_Testpred))
