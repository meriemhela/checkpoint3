#    Explore this dataset using what you have learned in data preprocessing and data visualization:

#importing the dataset : 
import pandas as pd
data = pd.read_csv(r"C:\Users\dell\Downloads\kc_house_data.csv",delimiter=',')
df=pd.DataFrame(data)


# Find missing values in the DataFrame:
missing_values = df.isnull().sum()
print("Missing values in each column:")
print()
print(missing_values)

print("*********************************************************************")



#      Split your dataset into a training set and a testing set:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import metrics

y=df[target]
x=df[features]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=30)


#        Apply Linear regression to your training set.

model=LinearRegression()
model.fit(x_train,y_train)
predicted=model.predict(x_test)

#        Plot the linear regression
plt.scatter(x,y,color="r")
plt.title("Linear Regression")
plt.xlabel()
plt.ylabel()
plt.plot(x,model.predict(x),color="k")
plt.show()

#         Measure the performance of linear regression using the testing set :
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_true, y_pred)
