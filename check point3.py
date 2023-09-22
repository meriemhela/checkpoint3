#    Explore this dataset using what you have learned in data preprocessing and data visualization:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#importing the dataset : 

data = pd.read_csv(r"C:\Users\dell\Downloads\kc_house_data.csv",delimiter=',')
df=pd.DataFrame(data)


# Find missing values in the DataFrame:
missing_values = df.isnull().sum()
print("Missing values in each column:")
print()
print(missing_values)

print("*********************************************************************")


#      Split your dataset into a training set and a testing set:
x=df[ ].values
y=df["price" ].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=30)

#("*********************************************************************")

#        Apply Linear regression to your training set.

# Create and train the multi-linear regression model
model=LinearRegression()
model.fit(x_train,y_train)

# Make predictions
y_pred =model.predict(x_test)

#        Plot the linear regression
plt.scatter(x,y,color="r")
plt.title("Linear Regression")
plt.xlabel()
plt.ylabel("price")
plt.plot(x,model.predict(x),color="k")
plt.show()

#         Measure the performance of linear regression using the testing set :

mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error: ",mse)
print("R-squared: ",r2)


print("*********************************************************************")

#         Apply multi-linear regression and compare it to the linear model:
X=df[].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and train the multi-linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
MSE = mean_squared_error(y_test, y_pred)
R2 = r2_score(y_test, y_pred)

print("Mean Squared Error: ",MSE)
print("R-squared: ",R2)

print("*********************************************************************")
#        Apply Polynomial regression and compare it to linear and multilinear regression

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X_train)

# Fit a linear regression model to the polynomial features
model.fit(X_poly, y_train)

# Make predictions
X_test_ = poly.fit_transform(X_test)
predicted = model.predict(X_test_)

# Evaluate the model
Mse = mean_squared_error(y_test, predicted)
R² = r2_score(y_test, predicted)

print("Mean Squared Error: ",Mse)
print("R-squared: ",R²)
