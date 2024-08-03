import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("/content/India Agriculture Crop Production.csv")

# Convert production units to tonnes
def change_units(row):
    if row['Production Units'] == 'Bales':
        return row['Production'] * 1.7
    elif row['Production Units'] == 'Nuts':
        return row['Production'] / 50
    else:
        return row['Production']

df['Production'] = df.apply(change_units, axis=1)

# Drop rows with missing 'Crop' values
df.dropna(subset=['Crop'], inplace=True)

# Filter relevant crops and keep the necessary columns
data = df[['State', 'District', 'Season', 'Area', 'Crop']]
data = data[data['Crop'].isin([
    'Rice', 'Maize', "Moong(Green Gram)", 'Urad', 'Groundnut', 'Wheat',
    'Rapeseed &Mustard', 'Sugarcane', 'Arhar/Tur', 'Potato', 'Onion', 'Gram',
    'Jowar', 'Dry Chilies', 'Bajra', "Peas & beans (Pulses)", 'Sunflower',
    'Banana', 'Coconut', 'Khesari', 'Small millets', 'Cotton(lint)', 'Masoor',
    'Turmeric', 'Barley', 'Linseed '
])]

# One-hot encode categorical variables
state = pd.get_dummies(data['State'], drop_first=True)
district = pd.get_dummies(data['District'], drop_first=True)
season = pd.get_dummies(data['Season'], drop_first=True)

# Concatenate features
new_data = pd.concat([data[['Area']], state, district, season], axis=1)

# Define features and target
X = new_data
Y = data['Crop']

# Handle missing values (if any)
if X.isna().sum().sum() > 0:
    imputer_X = SimpleImputer(strategy='mean')
    X = imputer_X.fit_transform(X)
else:
    X = X.values  # Ensure X is a numpy array for later steps

if Y.isna().sum() > 0:
    imputer_Y = SimpleImputer(strategy='most_frequent')
    Y = imputer_Y.fit_transform(Y.values.reshape(-1, 1)).ravel()
else:
    Y = Y.values  # Ensure Y is a numpy array for later steps

# Split the data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=50)

# Scale features
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
