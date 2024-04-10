#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from GCForest import gcForest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def load_large_csv(file_path):
    """
    Loads a CSV file into a DataFrame.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - DataFrame: The loaded data.
    """
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Preprocesses the data by removing unnecessary columns, extracting cancer types, and transposing the DataFrame.

    Parameters:
    - df (DataFrame): The input data.

    Returns:
    - tuple: A tuple containing the processed DataFrame and a dictionary of data filtered by cancer type.
    """
    # Drop the first column which is 'Unnamed: 0'
    df = df.drop(columns=['Unnamed: 0'])

    # Extract unique cancer types from column names
    cancer_types = df.columns.str.split('_').str[-1].unique()

    # Initialize a dictionary to store data filtered by cancer type
    filtered_data_by_cancer_type = {}
    for cancer_type in cancer_types:
        # Select columns corresponding to the current cancer type
        columns_for_type = df.columns[df.columns.str.endswith(cancer_type)]
        filtered_data_by_cancer_type[cancer_type] = df[columns_for_type]

    # Transpose the DataFrame for analysis (samples as rows, genes as columns)
    df = df.transpose()

    # Optionally, rename columns to gene indices for clarity
    df.columns = range(df.shape[1])

    return df, filtered_data_by_cancer_type

def calculate_fisher_ratio(filtered_data_by_cancer_type):
    """
    Calculates the Fisher ratio for each cancer type.

    Parameters:
    - filtered_data_by_cancer_type (dict): Data filtered by cancer type.

    Returns:
    - dict: Fisher scores for each cancer type.
    """
    fisher_scores = {}
    for cancer_type, data in filtered_data_by_cancer_type.items():
        # Calculate means and variances for genes, assuming data is transposed correctly
        means = data.mean(axis=1)
        variances = data.var(axis=1)

        # Calculate Fisher Score as the ratio of between-class variance to within-class variance
        fisher_score = means.var() / variances.mean()
        fisher_scores[cancer_type] = fisher_score

    return fisher_scores

file_path = '/Users/colt/Downloads/CPSC 571/wetransfer_init_data_design-csv_2024-03-12_2057/normalized_data.csv'
df = load_large_csv(file_path)

processed_df, filtered_data_by_cancer_type = preprocess_data(df)

# Calculate Fisher scores
fisher_scores = calculate_fisher_ratio(filtered_data_by_cancer_type)
print(fisher_scores)

# Create a DataFrame with Fisher scores
fisher_df = pd.DataFrame.from_dict(fisher_scores, orient='index', columns=['Fisher Score'])

# Merge the Fisher scores with the processed DataFrame
merged_df = pd.merge(processed_df, fisher_df, left_index=True, right_index=True)

# Sort the merged DataFrame by Fisher scores in descending order
sorted_df = merged_df.sort_values(by='Fisher Score', ascending=False)

# Select the top k features based on Fisher scores
k = 1000  # Select the top 1000 features
top_features = sorted_df.iloc[:, :-1].columns[:k]

# Extract cancer types from column headers for mapping
cancer_types = processed_df.index.str.split('_').str[-1]

# Map cancer types to numerical values
cancer_type_mapping = {'BRCA': 0, 'BRACA': 1, 'COCA': 2, 'KICA': 3, 'LECA': 4, 'LUCA': 5}
y = cancer_types.map(cancer_type_mapping).values

# Reset index of processed DataFrame to avoid confusion
processed_df.reset_index(drop=True, inplace=True)

# Use the selected top features as the feature set X
X = processed_df[top_features].values

# Setup K-Folds Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

for train_index, test_index in kf.split(X):
    # Split data into training and testing sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train and evaluate Deep Forest model
    gcf = gcForest(shape_1X=7, window=7, tolerance=0.0, min_samples_mgs=10, min_samples_cascade=7)
    gcf.fit(X_train, y_train)
    y_pred_gcf = gcf.predict(X_test)
    print("Deep Forest Accuracy:", accuracy_score(y_test, y_pred_gcf))

    # Train and evaluate Neural Network model
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(len(np.unique(y)), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    y_pred_nn = model.predict(X_test)
    y_pred_nn = np.argmax(y_pred_nn, axis=1)  # Convert probabilities to class labels
    print("Neural Network Accuracy:", accuracy_score(y_test, y_pred_nn))

    # Train and evaluate K-Nearest Neighbors model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))

# Reverse the cancer type mapping for interpretation
inv_cancer_type_mapping = {v: k for k, v in cancer_type_mapping.items()}

def print_accuracy_per_cancer_type(y_true, y_pred, inv_mapping):
    """
    Prints the accuracy for each cancer type based on true and predicted labels.

    Parameters:
    - y_true (array): True labels.
    - y_pred (array): Predicted labels.
    - inv_mapping (dict): Inverse mapping of cancer types to numerical labels.
    """
    # Convert numerical labels back to cancer type strings
    y_true_types = [inv_mapping[label] for label in y_true]
    y_pred_types = [inv_mapping[label] for label in y_pred]

    # Calculate and print accuracy for each cancer type
    for cancer_type in inv_mapping.values():
        # Find indices of true labels for the current cancer type
        indices = [i for i, x in enumerate(y_true_types) if x == cancer_type]
        if not indices:
            print(f"No samples for {cancer_type}.")
            continue

        # Subset true and predicted labels for the current cancer type
        y_true_subset = [y_true_types[i] for i in indices]
        y_pred_subset = [y_pred_types[i] for i in indices]

        # Calculate and print accuracy
        accuracy = accuracy_score(y_true_subset, y_pred_subset)
        print(f"Accuracy for {cancer_type}: {accuracy:.4f}")

print("Deep Forest Accuracy per Cancer Type:")
print_accuracy_per_cancer_type(y_test, y_pred_gcf, inv_cancer_type_mapping)

print("\nNeural Network Accuracy per Cancer Type:")
print_accuracy_per_cancer_type(y_test, y_pred_nn, inv_cancer_type_mapping)

print("\nKNN Accuracy per Cancer Type:")
print_accuracy_per_cancer_type(y_test, y_pred_knn, inv_cancer_type_mapping)
