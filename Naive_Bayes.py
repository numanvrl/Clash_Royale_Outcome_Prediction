import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import time

# Load the data
start_time = time.time()  # Record start time
data = pd.read_excel('player_battle_logs.xlsx')
end_time = time.time()  # Record end time
loading_time = end_time - start_time  # Calculate loading time
print("Data Loading Time:", loading_time, "seconds")

# Data cleaning
# For simplicity, let's assume there are no missing values or outliers in this example

# Feature engineering (if needed)
# You can create new features or transform existing ones here

# Split features and target variable
X = data.drop(columns=['Outcome'])
y = data['Outcome']

# Define categorical and numerical features
features = X.columns.tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features)
    ])

# Define the Random Forest model
rf_model = RandomForestClassifier()

# Create the Random Forest pipeline
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', rf_model)])

# Train-Test data split
start_time = time.time()  # Record start time
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
end_time = time.time()  # Record end time
splitting_time = end_time - start_time  # Calculate splitting time
print("Train-Test Splitting Time:", splitting_time, "seconds")

# Define the model
naive_bayes_model = GaussianNB()

# Create the full pipeline with Naive Bayes classifier
naive_bayes_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('classifier', naive_bayes_model)])

# Fit the Naive Bayes model
start_time = time.time()  # Record start time
naive_bayes_pipeline.fit(X_train, y_train)
end_time = time.time()  # Record end time
nb_training_time = end_time - start_time  # Calculate training time for Naive Bayes
print("Naive Bayes Model Training Time:", nb_training_time, "seconds")

# Predictions using Naive Bayes model
nb_predictions = naive_bayes_pipeline.predict(X_test)

# Calculate accuracy for Naive Bayes
nb_accuracy = accuracy_score(y_test, nb_predictions)
print("Naive Bayes Accuracy:", nb_accuracy)

# Calculate precision for Naive Bayes
nb_precision = precision_score(y_test, nb_predictions, average='weighted')
print("Naive Bayes Precision:", nb_precision)

# Calculate recall for Naive Bayes
nb_recall = recall_score(y_test, nb_predictions, average='weighted')
print("Naive Bayes Recall:", nb_recall)

# Calculate F1-score for Naive Bayes
nb_f1 = f1_score(y_test, nb_predictions, average='weighted')
print("Naive Bayes F1-score:", nb_f1)

# Calculate confusion matrix for Naive Bayes
nb_conf_matrix = confusion_matrix(y_test, nb_predictions)
print("Naive Bayes Confusion Matrix:")
print(nb_conf_matrix)
