import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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

# Initialize SVM model with RBF kernel
svm_model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42, probability=True)

# Train the model
print("Training SVM model...")
svm_model.fit(X_train, y_train)
print("Training complete!")

# Make predictions
y_pred = svm_model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM (RBF Kernel) Accuracy: {accuracy:.4f}")

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
plt.title("Confusion Matrix - SVM (RBF Kernel)")
plt.tight_layout()
plt.show()

# Get decision function values for one-vs-one classification
if len(encoder.classes_) > 2:  # Only applicable for multi-class
    # Get probability estimates
    y_proba = svm_model.predict_proba(X_test)
    
    # Create a plot to visualize prediction confidence
    plt.figure(figsize=(12, 6))
    
    # Plot confidence distribution for correct and incorrect predictions
    correct = y_test == y_pred
    plt.hist([np.max(y_proba[correct], axis=1), np.max(y_proba[~correct], axis=1)], 
             bins=20, label=['Correct predictions', 'Incorrect predictions'], alpha=0.7)
    plt.xlabel('Prediction Confidence (Max Probability)')
    plt.ylabel('Number of Samples')
    plt.title('Prediction Confidence Distribution - SVM')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Visualize decision boundary in reduced space (first two principal components)
    from sklearn.decomposition import PCA
    
    # Apply PCA to reduce to 2 dimensions for visualization
    pca = PCA(n_components=2)
    X_test_pca = pca.fit_transform(X_test)
    
    # Create a mesh grid to visualize decision boundaries
    h = 0.02  # step size in the mesh
    x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
    y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Train a new SVM on the PCA-transformed data
    svm_pca = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm_pca.fit(X_test_pca, y_test)
    
    # Predict on the mesh grid
    Z = svm_pca.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(12, 10))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    
    # Plot the test points
    scatter = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='viridis', 
                edgecolors='k', s=50, alpha=0.8)
    
    # Create a legend
    legend1 = plt.legend(*scatter.legend_elements(),
                        loc="upper right", title="Classes")
    plt.gca().add_artist(legend1)
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('SVM Decision Boundaries in PCA-Reduced Feature Space')
    plt.tight_layout()
    plt.show()