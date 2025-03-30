import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
df = pd.read_csv("features_30_sec.csv")

# Drop non-feature columns
df = df.drop(columns=["filename", "length"])

# Encode the labels
encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["label"])

# Get the label mapping for reference
label_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
# print(label_mapping)  # {'blues': 0, 'classical': 1, ...}

# Scale the features
scaler = StandardScaler()
X = df.drop(columns=["label"])
X_scaled = scaler.fit_transform(X)
y = df["label"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Initialize Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy:.4f}")

# Display detailed performance metrics
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.show()

# Feature importance analysis
feature_importance = rf_model.feature_importances_
feature_names = X.columns

# Sort features by importance
sorted_idx = np.argsort(feature_importance)[::-1]
sorted_features = feature_names[sorted_idx]
sorted_importance = feature_importance[sorted_idx]

# Plot feature importance
plt.figure(figsize=(12, 8))
plt.barh(range(20), sorted_importance[:20], align='center')
plt.yticks(range(20), sorted_features[:20])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Top 20 Important Features (Random Forest)')
plt.tight_layout()
plt.show()

# Compare feature importance distribution
plt.figure(figsize=(10, 6))
plt.hist(feature_importance, bins=20, alpha=0.7)
plt.xlabel('Feature Importance')
plt.ylabel('Number of Features')
plt.title('Distribution of Feature Importance')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()