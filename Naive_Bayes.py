import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Load the data
start_time = time.time()  # Record start time
data = pd.read_excel('player_battle_logs-outcome.xlsx')
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

# Plot Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(nb_conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot Performance Metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
scores = [nb_accuracy, nb_precision, nb_recall, nb_f1]

plt.figure(figsize=(10, 7))
plt.bar(metrics, scores, color=['blue', 'green', 'red', 'purple'])
plt.ylim(0, 1)
for i, score in enumerate(scores):
    plt.text(i, score + 0.01, round(score, 2), ha='center', va='bottom')
plt.title('Performance Metrics')
plt.xlabel('Metric')
plt.ylabel('Score')
plt.show()

# Plot Time Metrics
time_metrics = ['Data Loading', 'Train-Test Splitting', 'Model Training']
times = [loading_time, splitting_time, nb_training_time]

plt.figure(figsize=(10, 7))
plt.bar(time_metrics, times, color=['cyan', 'magenta', 'yellow'])
for i, time_value in enumerate(times):
    plt.text(i, time_value + 0.01, f'{time_value:.4f}', ha='center', va='bottom')
plt.title('Time Metrics')
plt.xlabel('Process')
plt.ylabel('Time (seconds)')
plt.show()