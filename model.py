import pandas as pd
import pickle
from flask import Flask, render_template, request, redirect, url_for
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
file_path = 'elect_data.csv'
election_data = pd.read_csv(file_path)

# Replace '\n' in column names with spaces
election_data.columns = election_data.columns.str.replace('\n', ' ')

# Replace 'Not available' with 0 in the 'CRIMINAL CASES' column
election_data['CRIMINAL CASES'].replace('Not Available', 0, inplace=True)

# Convert columns to numeric types
election_data['CRIMINAL CASES'] = pd.to_numeric(election_data['CRIMINAL CASES'], errors='coerce')

election_data['EDUCATION'] = election_data['EDUCATION'].replace(['Not Available','Others'],'Illiterate')
election_data['EDUCATION'] = election_data['EDUCATION'].replace(['Post Graduate\n'],'Post Graduate')
election_data['EDUCATION'] = election_data['EDUCATION'].replace(['5th Pass','8th Pass'],'Illiterate')

# Impute missing values in numeric columns with the median

numeric_columns = ['AGE', 'CRIMINAL CASES']
for col in numeric_columns:
    election_data[col].fillna(election_data[col].median(), inplace=True)

# Impute missing values in categorical columns with the most frequent category

categorical_columns = ['GENDER', 'CATEGORY', 'EDUCATION']
for col in categorical_columns:
    election_data[col].fillna(election_data[col].mode()[0], inplace=True)

# Impute missing values in 'SYMBOL' with 'Unknown'
election_data['SYMBOL'].fillna('Unknown', inplace=True)

# Clean and convert 'ASSETS' and 'LIABILITIES' columns
columns_to_fill = ['ASSETS', 'LIABILITIES']

for column in columns_to_fill:
    # Replace non-numeric characters with an empty string using regex
    election_data[column] = election_data[column].replace('[^0-9.]', '', regex=True)

    # Convert the column to numeric type
    election_data[column] = pd.to_numeric(election_data[column], errors='coerce')

    # Impute missing values with the median
    median_value = election_data[column].median()
    election_data[column].fillna(median_value, inplace=True)

# Drop specified columns
columns_to_drop = ['STATE', 'CONSTITUENCY', 'NAME', 'SYMBOL']
election_data.drop(columns_to_drop, axis=1, inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
for col in ['PARTY', 'GENDER', 'CATEGORY', 'EDUCATION']:
    election_data[col] = label_encoder.fit_transform(election_data[col])

# Split the data into features (X) and target variable (y)
X = election_data[['CATEGORY','AGE','CRIMINAL CASES','EDUCATION']]
y = election_data['WINNER']
print(X.columns)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train the Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
model = clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy}')

# Save the trained model using pickle
model_filename = 'election_model.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(clf, model_file)

