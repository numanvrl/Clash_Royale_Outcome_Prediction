import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.calibration import calibration_curve

# Load the data
data = pd.read_excel('player_battle_logs-output.xlsx')

# Check the distribution of Outcome classes
print("Distribution of Outcome classes:")
print(data['Outcome'].value_counts())

# Split features and target variable
X = data.drop(columns=['Outcome'])
y = data['Outcome']

# Encode categorical features
categorical_features = ['player_supportCard_1_rarity', 'opponent_supportCard_1_rarity',
                        'player_card_1_rarity', 'opponent_card_1_rarity',
                        'player_card_2_rarity', 'opponent_card_2_rarity',
                        'player_card_3_rarity', 'opponent_card_3_rarity',
                        'player_card_4_rarity', 'opponent_card_4_rarity',
                        'player_card_5_rarity', 'opponent_card_5_rarity',
                        'player_card_6_rarity', 'opponent_card_6_rarity',
                        'player_card_7_rarity', 'opponent_card_7_rarity',
                        'player_card_8_rarity', 'opponent_card_8_rarity']

# Define the rarity levels in the correct order
rarity_levels = ['common', 'rare', 'epic', 'legendary', 'champion']

# Replace NaNs with a placeholder value indicating 'unknown' rarity
X[categorical_features] = X[categorical_features].fillna('unknown')

# Initialize OrdinalEncoder with rarity levels
encoder = OrdinalEncoder(categories=[rarity_levels]*len(categorical_features), handle_unknown='use_encoded_value', unknown_value=-1)

# Encode categorical features
X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_features]))
X_encoded.columns = X_encoded.columns.astype(str)  # Convert column names to strings
X.drop(columns=categorical_features, inplace=True)
X = pd.concat([X, X_encoded], axis=1)

# Convert all column names to strings
X.columns = X.columns.astype(str)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Check the distribution of Outcome classes in training and testing sets
print("\nDistribution of Outcome classes in training set:")
print(y_train.value_counts())
print("\nDistribution of Outcome classes in testing set:")
print(y_test.value_counts())

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Replace missing values (NaNs) with the mean of each column
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train_scaled)
X_test_imputed = imputer.transform(X_test_scaled)

# Create KNN classifier with a specific value of k
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the classifier to the training data
knn.fit(X_train_imputed, y_train)

# Make predictions on the test data
y_pred = knn.predict(X_test_imputed)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
X_train_pca = pca.fit_transform(X_train_imputed)

# Plot the data points
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Plot of Data Points')
plt.legend(handles=scatter.legend_elements()[0], labels=['Loss', 'Win', 'Tie'], title='Outcome')
plt.grid(True)
plt.show()

# Convert multiclass labels to binary labels
y_test_binary = (y_test == 1)  # Convert all 'Win' (class 1) labels to True, others to False

# Get the probabilities for the positive class (class 1)
y_prob = knn.predict_proba(X_test_imputed)[:, 1]

# Compute calibration curve
prob_true, prob_pred = calibration_curve(y_test_binary, y_prob, n_bins=10)

# Plot calibration curve
plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker='o', color='blue', label='Calibration curve')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration Curve')
plt.legend(loc="upper left")
plt.show()
