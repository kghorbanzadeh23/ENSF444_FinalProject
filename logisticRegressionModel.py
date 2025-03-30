import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("features_30_sec.csv")

# Display the first few rows
# print(df.head())

# print(df.info())
# print(df.describe())
# print(df["label"].value_counts())  # Check genre distribution

df = df.drop(columns=["filename", "length"])
# print(df.isnull().sum().sum())  # Should print 0 if there are no missing values

encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["label"])

label_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
# print(label_mapping)  # {'blues': 0, 'classical': 1, ...}

scaler = StandardScaler()
X = df.drop(columns=["label"])
X_scaled = scaler.fit_transform(X)  # Converts to NumPy array
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)  # Check shapes of the splits

# Initialize Logistic Regression
log_reg = LogisticRegression(max_iter=1000)  # Increase iterations to ensure convergence

# Train the model
log_reg.fit(X_train, y_train)

# Make predictions
y_pred = log_reg.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {accuracy:.4f}")

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
plt.title("Confusion Matrix - Logistic Regression")
plt.show()
