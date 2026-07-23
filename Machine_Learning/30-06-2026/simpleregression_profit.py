import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load Dataset
data = pd.read_csv(r"D:/4062/PracticeMca/30-06-2026/linearregressiondataset.csv")

# Remove missing values
data = data.dropna()

# Independent and Dependent variables
x = data.iloc[:, 0:1]
y = data.iloc[:, 1]

print("X Data:")
print(x)

print("\nY Data:")
print(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=0
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Print model parameters
print("\nCoefficient:", model.coef_)
print("Intercept:", model.intercept_)

# Predict for new value
sample = pd.DataFrame([[20.27]], columns=x.columns)
prediction = model.predict(sample)

print("\nProfit for population 20.27 lakh:", prediction[0])

# Predict test data
y_pred = model.predict(X_test)

print("\nPredicted Values:")
print(y_pred)

# Evaluation
print("\nMean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 Score:", metrics.r2_score(y_test, y_pred))

# Plot
plt.scatter(x, y, color="blue")
plt.plot(x, model.predict(x), color="red")
plt.xlabel("Population")
plt.ylabel("Profit")
plt.title("Linear Regression")
plt.show()
