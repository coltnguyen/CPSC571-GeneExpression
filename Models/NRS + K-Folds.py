#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from GCForest import gcForest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors


def load_large_csv(file_path):
    """
    Load a large CSV file into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    return pd.read_csv(file_path)


def preprocess_data(df):
    """
    Preprocess the input DataFrame by removing unnecessary columns, splitting data by cancer type,
    and transposing the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        tuple: A tuple containing the preprocessed DataFrame and a dictionary of filtered data by cancer type.
    """
    df = df.drop(columns=['Unnamed: 0'])
    cancer_types = df.columns.str.split('_').str[-1].unique()
    filtered_data_by_cancer_type = {
        cancer_type: df[df.columns[df.columns.str.endswith(cancer_type)]]
        for cancer_type in cancer_types
    }
    df = df.transpose()
    df.columns = range(df.shape[1])
    return df, filtered_data_by_cancer_type


def calculate_neighborhood_rough_set(filtered_data_by_cancer_type, k=5):
    """
    Calculate the Neighborhood Rough Set for each cancer type.

    Args:
        filtered_data_by_cancer_type (dict): A dictionary of filtered data by cancer type.
        k (int): The number of nearest neighbors to consider.

    Returns:
        dict: A dictionary containing the Neighborhood Rough Set for each cancer type.
    """
    neighborhood_rough_sets = {}
    for cancer_type, data in filtered_data_by_cancer_type.items():
        X = data.values
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)
        rough_set_score = np.mean(distances[:, -1])
        neighborhood_rough_sets[cancer_type] = rough_set_score
    return neighborhood_rough_sets


def train_and_evaluate_models(X_train, X_test, y_train, y_test, y_test_binarized, inv_cancer_type_mapping, num_classes):
    """
    Train and evaluate multiple machine learning models on the given data.

    Args:
        X_train (np.ndarray): The training features.
        X_test (np.ndarray): The testing features.
        y_train (np.ndarray): The training labels.
        y_test (np.ndarray): The testing labels.
        y_test_binarized (np.ndarray): The binarized testing labels.
        inv_cancer_type_mapping (dict): The inverse mapping of cancer types to labels.
        num_classes (int): The number of unique classes.

    Returns:
        None
    """
    models = [
        ('Deep Forest', gcForest(shape_1X=7, window=7, tolerance=0.0, min_samples_mgs=10, min_samples_cascade=7)),
        ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('Neural Network', create_neural_network(X_train.shape[1], num_classes)),
        ('KNN', KNeighborsClassifier(n_neighbors=5))
    ]

    for model_name, model in models:
        if model_name == 'Neural Network':
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
            y_pred = model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_classes = y_pred

        print(f"{model_name} Accuracy:", accuracy_score(y_test, y_pred_classes))
        print(f"{model_name} Precision:", precision_score(y_test, y_pred_classes, average='weighted'))
        print(f"{model_name} Recall:", recall_score(y_test, y_pred_classes, average='weighted'))
        print(f"{model_name} F1-score:", f1_score(y_test, y_pred_classes, average='weighted'))

        if model_name == 'Neural Network':
            print(f"{model_name} AUC:", roc_auc_score(y_test_binarized, y_pred, multi_class='ovr'))
        else:
            y_pred_proba = model.predict_proba(X_test)
            print(f"{model_name} AUC:", roc_auc_score(y_test_binarized, y_pred_proba, multi_class='ovr'))

        print(f"\n{model_name} Accuracy per Cancer Type:")
        print_accuracy_per_cancer_type(y_test, y_pred_classes, inv_cancer_type_mapping)
        print()


def create_neural_network(input_shape, num_classes):
    """
    Create a neural network model using Keras.

    Args:
        input_shape (int): The shape of the input features.
        num_classes (int): The number of output classes.

    Returns:
        keras.Model: The created neural network model.
    """
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def print_accuracy_per_cancer_type(y_true, y_pred, inv_mapping):
    """
    Print the accuracy for each cancer type.

    Args:
        y_true (np.ndarray): The true labels.
        y_pred (np.ndarray): The predicted labels.
        inv_mapping (dict): The inverse mapping of labels to cancer types.

    Returns:
        None
    """
    y_true_types = [inv_mapping[label] for label in y_true]
    y_pred_types = [inv_mapping[label] for label in y_pred]

    for cancer_type in inv_mapping.values():
        indices = [i for i, x in enumerate(y_true_types) if x == cancer_type]
        if not indices:
            print(f"No samples for {cancer_type}.")
            continue

        y_true_subset = [y_true_types[i] for i in indices]
        y_pred_subset = [y_pred_types[i] for i in indices]

        accuracy = accuracy_score(y_true_subset, y_pred_subset)
        print(f"Accuracy for {cancer_type}: {accuracy:.4f}")


if __name__ == '__main__':
    # Load the CSV file
    file_path = '/Users/colt/Downloads/CPSC 571/wetransfer_init_data_design-csv_2024-03-12_2057/normalized_data.csv'
    df = load_large_csv(file_path)

    # Preprocess the data
    processed_df, filtered_data_by_cancer_type = preprocess_data(df)

    # Calculate Neighborhood Rough Set for each cancer type
    neighborhood_rough_sets = calculate_neighborhood_rough_set(filtered_data_by_cancer_type)
    print(neighborhood_rough_sets)

    # Merge Neighborhood Rough Set scores with the processed DataFrame and sort by score
    neighborhood_rough_set_df = pd.DataFrame.from_dict(neighborhood_rough_sets, orient='index', columns=['Neighborhood Rough Set Score'])
    merged_df = pd.merge(processed_df, neighborhood_rough_set_df, left_index=True, right_index=True)
    sorted_df = merged_df.sort_values(by='Neighborhood Rough Set Score', ascending=False)

    # Select the top k features
    k = 1000
    # Select the top k features based on Neighborhood Rough Set score
    top_features = sorted_df.iloc[:, :-1].columns[:k]

    # Map cancer types to numerical labels
    cancer_types = processed_df.index.str.split('_').str[-1]
    cancer_type_mapping = {'BRCA': 0, 'BRACA': 1, 'COCA': 2, 'KICA': 3, 'LECA': 4, 'LUCA': 5}
    y = cancer_types.map(cancer_type_mapping).values

    # Reset index of the processed DataFrame
    processed_df.reset_index(drop=True, inplace=True)

    # Select the top features from the processed DataFrame
    X = processed_df[top_features].values

    # Initialize KFold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Create an inverse mapping of cancer type labels to names
    inv_cancer_type_mapping = {v: k for k, v in cancer_type_mapping.items()}

    # Perform cross-validation
    for train_index, test_index in kf.split(X):
        # Split the data into training and testing sets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Binarize the testing labels for multi-class AUC calculation
        y_test_binarized = label_binarize(y_test, classes=np.unique(y))

        # Get the number of unique classes
        num_classes = len(np.unique(y))

        # Train and evaluate models
        train_and_evaluate_models(X_train, X_test, y_train, y_test, y_test_binarized, inv_cancer_type_mapping, num_classes)
