import pandas as pd

# Load the crop production data
crop_data = pd.read_csv("/content/India Agriculture Crop Production.csv")

# Load the rainfall data
rainfall_data = pd.read_csv("/content/Copy of rainfall.csv")

# Rename 'Crop_Year' to 'Year' for consistency
rainfall_data.rename(columns={'Crop_Year': 'Year'}, inplace=True)
# Function to clean the Year column
def clean_year(year):
    if isinstance(year, str):
        return int(year.split('-')[0])
    return int(year)

# Clean and convert the 'Year' column in both datasets
crop_data['Year'] = crop_data['Year'].apply(clean_year)
rainfall_data['Year'] = rainfall_data['Year'].apply(clean_year)
# Create a dictionary from rainfall data for easy lookup
rainfall_dict = rainfall_data.set_index('Year')['rain_in_mm'].to_dict()

# Add a new column for rainfall in the crop production data
crop_data['rain_in_mm'] = crop_data['Year'].map(rainfall_dict)

# Select relevant columns and remove duplicates and missing values
data = crop_data[['Year', 'rain_in_mm']].drop_duplicates().dropna()
# Prepare features and target
X = data[['Year']]
y = data['rain_in_mm']
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=50)
model.fit(x_train, y_train)

# Predict rainfall on the test set
y_pred = model.predict(x_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R^2 Score: {r2:.2f}")


import matplotlib.pyplot as plt

# Plot actual vs predicted rainfall
plt.figure(figsize=(10, 6))
plt.scatter(x_test, y_test, color='blue', label='Actual Rainfall')
plt.scatter(x_test, y_pred, color='red', label='Predicted Rainfall')
plt.xlabel('Year')
plt.ylabel('Rainfall (mm)')
plt.title('Actual vs Predicted Rainfall')
plt.legend()
plt.show()
