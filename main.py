import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load your datasets
a = pd.read_csv(r"C:\Users\Niharika Kashyap\Downloads\Copy of Crop Data.csv")
b = pd.read_csv(r"C:\Users\Niharika Kashyap\Downloads\Copy of Chemicals.csv")
c=pd.read_csv(r"C:\Users\Niharika Kashyap\Downloads\crop damage.csv")
print(a.info(), b.info())

a.dropna(subset=['Production'], inplace=True)

a['Season'] = a['Season'].str.strip()
b['Season'] = b['Season'].str.strip()
c['Season'] = c['Season'].str.strip()



concatenated_data = pd.concat([a, b, c], axis=0, ignore_index=True)
concatenated_data.fillna({
    'State_Name': 'Unknown',
    'District_Name': 'Unknown',
    'Crop_Year': 0,
    'Season': 'Unknown',
    'Crop': 'Unknown',
    'Area': 0.0,
    'Production': 0.0,
    'Estimated_Insects_Count': 0,
    'Crop_Type': 'Unknown',
    'Soil_Type': 'Unknown',
    'Pesticide_Use_Category': 'Unknown',
    'Number_Doses_Week': 0,
    'Number_Weeks_Used': 0,
    'Crop_Damage': 'Unknown'
}, inplace=True)
concatenated_data.drop_duplicates(inplace=True)
concatenated_data['Crop_Year'] = concatenated_data['Crop_Year'].astype(int)
concatenated_data['Area'] = concatenated_data['Area'].astype(float)
concatenated_data['Production'] = concatenated_data['Production'].astype(float)

X = concatenated_data.drop(columns='Production')
y = concatenated_data['Production']


X = pd.get_dummies(X, drop_first=True)
numerical_features = ['Crop_Year', 'Area', 'Estimated_Insects_Count', 'Number_Doses_Week', 'Number_Weeks_Used']
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print(X_train.head())

print(X_test.head())



print(concatenated_data)
