import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset (You need to have a dataset with 'URL' and 'label' columns)
# Example: Replace 'dataset.csv' with your actual dataset file name
data = pd.read_csv('dataset.csv')

# Step 2: Preprocess the dataset (clean data, handle missing values, extract features)
# You may need to write custom functions to extract features from URLs

# Assuming you have extracted features and stored them in the 'features' DataFrame
# features = ...

# Step 3: Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(features, data['label'], test_size=0.2, random_state=42)

# Step 4: Train the Gradient Boosting model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Step 6: Predict if a new URL is phishing or not
# Assuming you have a new URL stored in the 'new_url' variable
# new_url_features = extract_features(new_url)
# prediction = model.predict(new_url_features)

# Depending on your actual implementation, you can use 'prediction' to determine if the URL is phishing or not.

