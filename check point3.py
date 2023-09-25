import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Import the dataset
data = pd.read_csv(r"C:\Users\dell\Downloads\kc_house_data.csv", delimiter=',')
df = pd.DataFrame(data)

# Find missing values in the DataFrame
missing_values = df.isnull().sum()
print("Missing values in each column:")
print()
print(missing_values) 


# Split your dataset into a training set and a testing set
x = df[["sqft_living"]]  
y = df["price"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)

# Apply Linear regression to your training set
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)

# Plot the linear regression
plt.scatter(x_test, y_test, color="r")
plt.title("Linear Regression")
plt.xlabel(" ")
plt.ylabel("Price")
plt.plot(x_test, y_pred, color="k")
plt.show()

# Measure the performance of linear regression using the testing set
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error (Linear Regression): ", mse)
print("R-squared (Linear Regression): ", r2)

# Apply multi-linear regression and compare it to the linear model
X = df[["sqft_living", "bedrooms", "bathrooms", "floors", "view", "condition", "grade", "yr_built", "yr_renovated", "lat", "long", "sqft_living15"]]  # Use multiple features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

MSE = mean_squared_error(y_test, y_pred)
R2 = r2_score(y_test, y_pred)

print("Mean Squared Error (Multi-Linear Regression): ", MSE)
print("R-squared (Multi-Linear Regression): ", R2)

# Apply Polynomial regression and compare it to linear and multilinear regression
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X_train)
model.fit(X_poly, y_train)
X_test_ = poly.transform(X_test)
predicted = model.predict(X_test_)

Mse = mean_squared_error(y_test, predicted)
RR = r2_score(y_test, predicted)

print("Mean Squared Error (Polynomial Regression): ", Mse)
print("R-squared (Polynomial Regression): ", RR)
