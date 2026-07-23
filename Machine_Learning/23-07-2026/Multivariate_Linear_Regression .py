import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load Dataset
data = pd.read_excel("D:/4062/PracticeMca/30-06-2026/Multivariate Linear Regression Dataset.xlsx")

print(data.head())

# Features
X = data[['Square Feet', 'Number of Bed Rooms']]

# Target
y = data['Price of House']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Predicted Price")
print(y_pred)

# Accuracy
print("R2 Score :", r2_score(y_test, y_pred))
print("MAE :", mean_absolute_error(y_test, y_pred))

# Predict New House
new_house = [[2000, 3]]

price = model.predict(new_house)

print("Predicted House Price =", price[0])
