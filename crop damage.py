import pandas as pd

# Load the new dataset
soil_data = pd.read_csv("/content/chemicals.csv")

# Display the first few rows to understand the data
print(soil_data.head())
# One-hot encode categorical variables, including 'soil type', 'season', 'szn', and 'item'
soil_data_encoded = pd.get_dummies(soil_data, columns=['Soil_Type', 'Season', 'Szn', 'Item'], drop_first=True)

# Check the new DataFrame after encoding
print(soil_data_encoded.head())
# Define features and target
X = soil_data_encoded.drop(columns=['Crop_Damage'])
y = soil_data_encoded['Crop_Damage']


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=50)
model.fit(x_train, y_train)

# Predict crop damage on the test set
y_pred = model.predict(x_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

import joblib
joblib.dump(model, 'crop_damage_model.pkl')import joblib
import pandas as pd

# Load the trained model
model = joblib.load('crop_damage_model.pkl')

# Function to get user input
def get_user_input():
    soil_type = input("Enter soil type: ")
    season = input("Enter crop season (e.g., Kharif, Rabi): ")
    szn = input("Enter season (e.g., Monsoon, Summer, Winter): ")
    item = input("Enter item used (e.g., Insecticide, Pesticide): ")
    doses = int(input("Enter number of doses: "))

    # Prepare the input DataFrame
    input_data = {
        'soil_type': [soil_type],
        'season': [season],
        'szn': [szn],
        'item': [item],
        'no_of_doses': [doses]
    }
    input_df = pd.DataFrame(input_data)

    # One-hot encode the input DataFrame
    input_encoded = pd.get_dummies(input_df, columns=['soil_type', 'season', 'szn', 'item'], drop_first=True)

    # Ensure all expected columns are present
    model_columns = model.feature_names_in_
    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    input_encoded = input_encoded[model_columns]

    return input_encoded

# Function to make prediction
def predict_crop_damage(input_data):
    prediction = model.predict(input_data)[0]
    return prediction

# Main function
if __name__ == "__main__":
    user_input = get_user_input()
    result = predict_crop_damage(user_input)
    print(f"Predicted crop damage: {result}")

