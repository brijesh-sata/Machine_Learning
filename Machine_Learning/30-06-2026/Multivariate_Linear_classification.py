import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
data = pd.read_excel("D:/4062/PracticeMca/30-06-2026/Multivariate Linear Regression Dataset.xlsx")

# Convert Price into Classes
median_price = data["Price of House"].median()

data["House_Class"] = (data["Price of House"] >= median_price).astype(int)

print(data.head())

# Features
X = data[['Square Feet', 'Number of Bed Rooms']]

# Target
y = data['House_Class']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Accuracy =", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

# Predict New House
new_house = [[2000, 3]]

result = model.predict(new_house)

if result[0] == 1:
    print("Expensive House")
else:
    print("Cheap House")
