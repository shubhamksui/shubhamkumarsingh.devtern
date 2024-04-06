# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('house_price_dataset.csv')

# Data preprocessing
X = data.drop('Price', axis=1)  # Features
y = data['Price']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Model evaluation
train_preds = model.predict(X_train)
train_rmse = mean_squared_error(y_train, train_preds, squared=False)
print(f'Training RMSE: {train_rmse}')

test_preds = model.predict(X_test)
test_rmse = mean_squared_error(y_test, test_preds, squared=False)
print(f'Testing RMSE: {test_rmse}')

# Example prediction
example_data = pd.DataFrame({'Feature1': [value1], 'Feature2': [value2], ...})  # Replace value1, value2, ... with actual values
predicted_price = model.predict(example_data)
print(f'Predicted price: {predicted_price[0]}')
