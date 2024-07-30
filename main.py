import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load your datasets
crop_data = pd.read_csv(r"C:\Users\Niharika Kashyap\Downloads\Copy of Crop Data.csv")
rainfall_data = pd.read_csv(r"C:\Users\Niharika Kashyap\Downloads\Copy of rainfall.csv")
chem_data = pd.read_csv(r"C:\Users\Niharika Kashyap\Downloads\chemicals.csv")
pest_data=pd.read_csv(r"C:\Users\Niharika Kashyap\Downloads\pest.csv")

# Merge datasets
data = pd.merge(crop_data, rainfall_data, how='inner', left_on=['Crop_Year', 'STATE_NAME'], right_on=['Crop_Year', 'STATE_NAME'])
data = pd.merge(data, chem_data, how='inner', on=['Season'])
#data = pd.merge(data, pest_data, how='inner', on=['Crop_Year'])


data.drop(columns=['date', 'State_Name', 'District_Name'], inplace=True, errors='ignore')  # Use errors='ignore' to handle cases where columns might not exist
data.fillna(method='ffill', inplace=True)  # Forward fill missing values if applicable


data['Total_Rainfall'] = data.groupby('STATE_NAME')['rain_in_mm'].transform('sum')

plt.figure(figsize=(10, 6))
data.groupby('Crop_Year')['Total_Rainfall'].mean().plot()
plt.title('Average Annual Rainfall Over Years')
plt.xlabel('Year')
plt.ylabel('Rainfall (mm)')
plt.show()

grouped_data = data.groupby(['Soil_Type', 'Crop_Damage'])['Estimated_Insects_Count'].mean().reset_index()

# Plot the data using seaborn
plt.figure(figsize=(14, 8))
sns.barplot(x='Crop_Damage', y='Estimated_Insects_Count', hue='Soil_Type', data=grouped_data)
plt.title('Mean Pesticide Usage by Crop Damage Category and Soil Type')
plt.xlabel('Crop Damage Category')
plt.ylabel('Mean Pesticide Usage')
plt.legend(title='Soil Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

plt.figure(figsize=(10, 6))
data.groupby('Crop_Year')['Estimated_Insects_Count'].mean().plot()
plt.title('Average Pesticide Usage Over Years')
plt.xlabel('Year')
plt.ylabel('Pesticide Usage')
plt.show()

# Analyze the relationship between pesticide usage and crop damage
plt.figure(figsize=(10, 6))
plt.scatter(data['Estimated_Insects_Count'], data['Crop_Damage'])
plt.xlabel('Pesticide Usage')
plt.ylabel('Crop Damage')
plt.title('Pesticide Usage vs Crop Damage')
plt.show()

# Define threshold for sustainable pesticide use
pesticide_threshold = data['Estimated_Insects_Count'].mean()

# Recommend sustainable practices based on pesticide usage
data['Sustainable_Practice'] = data['Value'].apply(lambda x: 'Reduce Pesticide' if x > pesticide_threshold else 'Maintain Current Practice')

# Analyze recommended practices by soil type
sustainable_practices_by_soil = data.groupby('Soil_Type')['Sustainable_Practice'].value_counts().unstack().fillna(0)
sustainable_practices_by_soil.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Sustainable Practices by Soil Type')
plt.xlabel('Soil Type')
plt.ylabel('Count of Recommendations')
plt.show()


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Split the data into features (X) and target (y)
X = data[['Value', 'Soil_Type', 'Area']]  # Add other relevant features
y = data['Crop_Damage']

# Convert categorical variables to numeric
X = pd.get_dummies(X, columns=['Soil_Type'], drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'RMSE: {rmse}')

# Visualize the actual vs predicted crop damage
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.xlabel('Actual Crop Damage')
plt.ylabel('Predicted Crop Damage')
plt.title('Actual vs Predicted Crop Damage')
plt.show()


# Assuming crop damage data affects supply chain efficiency
data['Supply_Chain_Efficiency'] = (data['Production'] - data['Crop_Damage']) / data['Production']

# Visualize supply chain efficiency
plt.figure(figsize=(10, 6))
data.groupby('Crop_Year')['Supply_Chain_Efficiency'].mean().plot()
plt.title('Average Supply Chain Efficiency Over Years')
plt.xlabel('Year')
plt.ylabel('Efficiency')
plt.show()
