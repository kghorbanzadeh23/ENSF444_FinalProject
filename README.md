# Music Genre Classification

This project implements four different machine learning models to classify music genres based on audio features extracted from 30-second song clips. The models include:
- Logistic Regression
- K-Nearest Neighbors
- Random Forest
- Support Vector Machine (SVM)

## Prerequisites

To run the models, you'll need the following Python libraries:
```bash
pip install numpy pandas matplotlib scikit-learn seaborn
```

## Dataset

The dataset (`features_30_sec.csv`) contains audio features extracted from 30-second song clips including:
- Spectral features (centroid, bandwidth, rolloff)
- Rhythmic features (tempo, zero-crossing rate)
- MFCC (Mel-frequency cepstral coefficients)
- Chroma features
- RMS energy

## IMPORTANT NOTE
To see the additional visualizations for each model, by exiting out of the tab created for the first one which displays, then waiting a couple seconds, the next will appear

## Running the Models

### 1. Logistic Regression Model

This model applies multinomial logistic regression for genre classification:

```bash
python logisticRegressionModel.py
```

The output includes:
- Classification accuracy
- Detailed classification report
- Confusion matrix visualization

### 2. K-Nearest Neighbors Model

The KNN model includes hyperparameter tuning to find the optimal value of k and distance metric:

```bash
python knnModel.py
```

The output includes:
- Optimized model parameters
- Cross-validation accuracy
- Classification accuracy on test set
- Confusion matrix visualization
- Elbow method plots for different distance metrics

### 3. Random Forest Model

This ensemble model builds multiple decision trees for classification:

```bash
python randomForestModel.py
```

The output includes:
- Classification accuracy
- Detailed classification report
- Confusion matrix visualization
- Feature importance visualization (top 20 features)
- Distribution of feature importance

### 4. SVM Model

The Support Vector Machine model uses RBF kernel to handle non-linear relationships:

```bash
python svmModel.py
```

The output includes:
- Classification accuracy
- Detailed classification report
- Confusion matrix visualization
- Prediction confidence distribution
- PCA-based decision boundary visualization

## Comparing Model Performance

All four models use the same train-test split (80-20) with identical random seed (42) for fair comparison. 
Each model generates performance metrics that can be compared to determine which approach works best for 
music genre classification.

To compare the models directly, look at:
1. Overall accuracy
2. F1 scores for each genre
3. Confusion matrices to understand which genres are commonly confused

