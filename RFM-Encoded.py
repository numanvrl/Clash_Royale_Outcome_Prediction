import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
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
categorical_features = ['player_supportCard_1_rarity', 'opponent_supportCard_1_rarity',
                        'player_card_1_rarity', 'opponent_card_1_rarity',
                        'player_card_2_rarity', 'opponent_card_2_rarity',
                        'player_card_3_rarity', 'opponent_card_3_rarity',
                        'player_card_4_rarity', 'opponent_card_4_rarity',
                        'player_card_5_rarity', 'opponent_card_5_rarity',
                        'player_card_6_rarity', 'opponent_card_6_rarity',
                        'player_card_7_rarity', 'opponent_card_7_rarity',
                        'player_card_8_rarity', 'opponent_card_8_rarity']  # Add all categorical features here
numerical_features = ['player_kingTowerHitPoints', 'player_leftPrincessTowerHitPoints', 'player_rightPrincessTowerHitPoints',
                      'opponent_kingTowerHitPoints', 'opponent_leftPrincessTowerHitPoints', 'opponent_rightPrincessTowerHitPoints',
                      'player_elixirLeaked', 'opponent_elixirLeaked',
                      'player_supportCard_1_level', 'opponent_supportCard_1_level',
                      'player_card_1_level', 'opponent_card_1_level',
                      'player_card_2_level', 'opponent_card_2_level',
                      'player_card_3_level', 'opponent_card_3_level',
                      'player_card_4_level', 'opponent_card_4_level',
                      'player_card_5_level', 'opponent_card_5_level',
                      'player_card_6_level', 'opponent_card_6_level',
                      'player_card_7_level', 'opponent_card_7_level',
                      'player_card_8_level', 'opponent_card_8_level',
                      'player_card_1_elixirCost', 'opponent_card_1_elixirCost',
					  'player_card_2_elixirCost', 'opponent_card_2_elixirCost',
					  'player_card_3_elixirCost', 'opponent_card_3_elixirCost',
					  'player_card_4_elixirCost', 'opponent_card_4_elixirCost',
					  'player_card_5_elixirCost', 'opponent_card_5_elixirCost',
					  'player_card_6_elixirCost', 'opponent_card_6_elixirCost',
					  'player_card_7_elixirCost', 'opponent_card_7_elixirCost',
					  'player_card_8_elixirCost', 'opponent_card_8_elixirCost', ]  # Add all numerical features here

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OrdinalEncoder(), categorical_features)
    ])

# Define the model
model = RandomForestClassifier()

# Create the full pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', model)])

# Train-Test data
start_time = time.time()  # Record start time
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
end_time = time.time()  # Record end time
splitting_time = end_time - start_time  # Calculate splitting time
print("Train-Test Splitting Time:", splitting_time, "seconds")

# Fit the model
start_time = time.time()  # Record start time
pipeline.fit(X_train, y_train)
end_time = time.time()  # Record end time
training_time = end_time - start_time  # Calculate training time
print("Model Training Time:", training_time, "seconds")

predictions = pipeline.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Calculate precision
precision = precision_score(y_test, predictions, average='weighted')
print("Precision:", precision)

# Calculate recall
recall = recall_score(y_test, predictions, average='weighted')
print("Recall:", recall)

# Calculate F1-score
f1 = f1_score(y_test, predictions, average='weighted')
print("F1-score:", f1)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Get feature importances from the trained model
feature_importances = pipeline.named_steps['classifier'].feature_importances_

# Get one-hot encoder feature names
feature_names_cat = pipeline.named_steps['preprocessor'].named_transformers_['cat'].categories_

# Combine feature names
all_feature_names = list(numerical_features)
for i, feature in enumerate(categorical_features):
    for level in feature_names_cat[i]:
        all_feature_names.append(f"{feature}_{level}")

plt.figure(figsize=(12, 8))  # Increase figure size
plt.barh(all_feature_names, feature_importances)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.xticks(fontsize=10)  # Adjust font size of x-axis ticks
plt.yticks(fontsize=8)  # Adjust font size of y-axis ticks
plt.tight_layout()  # Automatically adjust subplot parameters to give specified padding
plt.show()
