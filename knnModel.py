import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
df = pd.read_csv("features_30_sec.csv")

# Drop non-feature columns (filename and length, if present)
df = df.drop(columns=["filename", "length"], errors='ignore')

# Encode the labels as integers
encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["label"])

# Separate features (X) and target labels (y)
X = df.drop(columns=["label"])
y = df["label"]

# Scale the features to improve k-NN performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_neighbors': range(1, 21),  # Testing k values from 1 to 20
    'metric': ['euclidean', 'manhattan', 'minkowski']  # Different distance metrics
}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Retrieve the best model from GridSearch
best_knn = grid_search.best_estimator_
best_k = grid_search.best_params_['n_neighbors']
print(f"Best KNN Parameters: {grid_search.best_params_}")
print(f"Best k: {best_k}")

# Perform cross-validation to evaluate model performance
cv_scores = cross_val_score(best_knn, X_train, y_train, cv=5)
print(f"Cross-Validation Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# Train the best model on the full training set
best_knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_knn.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Optimized KNN Accuracy: {accuracy:.4f}")

# Display detailed classification metrics
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# --- Confusion Matrix ---
# Plot the confusion matrix using Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Optimized KNN")
plt.tight_layout()
plt.show()

# --- Multi-Metric Elbow Method ---
metrics = ['euclidean', 'manhattan', 'minkowski']
k_values = list(range(1, 21))
metric_scores = {}

# Calculate cross-validation scores for each metric
for metric in metrics:
    scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k, metric=metric), 
                              X_train, y_train, cv=5).mean() for k in k_values]
    metric_scores[metric] = scores

# Plot the elbow graph for each metric with transparency and distinct markers
plt.figure(figsize=(12, 8))

# Marker styles for better differentiation
markers = {'euclidean': 'o', 'manhattan': 's', 'minkowski': 'D'}

for metric, scores in metric_scores.items():
    plt.plot(k_values, scores, marker=markers[metric], label=f"{metric.capitalize()}", alpha=0.8)

plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Cross-Validation Accuracy")
plt.title("Elbow Method for Different Distance Metrics")
plt.xticks(k_values)
plt.grid(True)
plt.legend()
plt.show()

# --- HeatMap ---
# Display Classification Report Heatmap
report_dict = classification_report(y_test, y_pred, target_names=encoder.classes_, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()
plt.figure(figsize=(8, 6))
sns.heatmap(df_report.iloc[:-1, :-1], annot=True, cmap="Blues", fmt=".2f")
plt.title("Classification Report Heatmap")
plt.show()
