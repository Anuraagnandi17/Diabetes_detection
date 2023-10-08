import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/diabetes/diabetes.data"
names = ["Age", "Sex", "BMI", "BP", "S1", "S2", "S3", "S4", "S5", "S6", "Target"]
data = pd.read_csv(url, delimiter="\t", header=None, names=names)

# Split the data into features (X) and target (y)
X = data.drop(columns=["Target"])
y = data["Target"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the mean squared error and R-squared score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Plot the predicted vs. actual values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Diabetes Progression")
plt.ylabel("Predicted Diabetes Progression")
plt.title("Actual vs. Predicted Diabetes Progression")
plt.show()
