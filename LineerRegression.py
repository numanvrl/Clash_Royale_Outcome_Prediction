import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import seaborn as sns

# Load the data from Excel file
data = pd.read_excel('player_battle_logs-outcome.xlsx')

# Split features and target variable
X = data.drop(columns=['Outcome'])
y = data['Outcome']

features = X.columns.tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features),  # Scale numerical features
    ])

# Define the model
model = LinearRegression()  # Linear Regression model

# Create the full pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', model)])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the model
pipeline.fit(X_train, y_train)

# Make predictions
predictions = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Convert predictions to binary outcome (0 or 1) for confusion matrix and classification metrics
binary_predictions = [1 if pred >= 0.5 else 0 for pred in predictions]

# Compute confusion matrix and other metrics
conf_matrix = confusion_matrix(y_test, binary_predictions)
accuracy = accuracy_score(y_test, binary_predictions)
recall = recall_score(y_test, binary_predictions, average='macro')
precision = precision_score(y_test, binary_predictions, average='macro')
f1 = f1_score(y_test, binary_predictions, average='macro')

print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("F1 Score:", f1)

# Plotting the evaluation metrics
metrics = {'Mean Squared Error': mse, 'R-squared': r2, 'Accuracy': accuracy, 'Recall': recall, 'Precision': precision, 'F1 Score': f1}

plt.figure(figsize=(12, 6))
plt.bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'orange', 'red', 'purple', 'cyan'])
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.title('Evaluation Metrics')
plt.ylim([0, 1.1 * max(metrics.values())])
plt.show()

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
