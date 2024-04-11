#!/usr/bin/env python3

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras_tuner import RandomSearch
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

def calculate_fisher_ratio(filtered_data_by_cancer_type):
    """
    Calculate the Fisher ratio for each cancer type, which is a measure of variance between classes
    divided by the variance within classes.

    Args:
        filtered_data_by_cancer_type (dict): A dictionary of filtered data by cancer type.

    Returns:
        dict: A dictionary containing the Fisher ratio for each cancer type.
    """
    fisher_scores = {}
    for cancer_type, data in filtered_data_by_cancer_type.items():
        means = data.mean(axis=1)
        variances = data.var(axis=1)
        fisher_score = means.var() / variances.mean()
        fisher_scores[cancer_type] = fisher_score
    return fisher_scores

def calculate_nrs_score(X, y, k=5):
    """
    Calculate the Neighborhood Rough Set (NRS) score for each feature, which is a measure of feature importance
    based on the concept of rough sets in the context of nearest neighbors.

    Args:
        X (np.ndarray): The feature matrix.
        y (np.ndarray): The target labels.
        k (int): The number of nearest neighbors to consider.

    Returns:
        np.ndarray: The NRS scores for each feature.
    """
    n_samples, n_features = X.shape
    nrs_scores = np.zeros(n_features)
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    for i in range(n_samples):
        neighbor_labels = y[indices[i]]
        upper_approx = np.sum(neighbor_labels == y[i])
        lower_approx = np.sum(neighbor_labels != y[i])
        diff = X[i] - X[indices[i]]
        nrs_scores += np.abs(diff).sum(axis=0) * (upper_approx - lower_approx)
    return nrs_scores / n_samples

def build_model(hp):
    """
    Build a neural network model with hyperparameters defined by Keras Tuner.

    Args:
        hp (HyperParameters): Hyperparameters object to define the model architecture dynamically.

    Returns:
        keras.Model: The constructed neural network model.
    """
    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
                        activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate_model(X_train, X_test, y_train, y_test, cancer_type_mapping):
    """
    Train and evaluate the neural network model using Keras Tuner to find the best hyperparameters.
    Additionally, print the accuracy for each cancer type.

    Args:
        X_train (np.ndarray): The training feature set.
        X_test (np.ndarray): The testing feature set.
        y_train (np.ndarray): The training labels.
        y_test (np.ndarray): The testing labels.
        cancer_type_mapping (dict): Mapping of cancer types to numerical labels.

    Returns:
        None
    """
    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=3,
        directory='my_dir',
        project_name='hparam_tuning'
    )

    tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    best_model = tuner.get_best_models(num_models=1)[0]
    y_pred = best_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Calculate overall metrics
    print("Neural Network Accuracy:", accuracy_score(y_test, y_pred_classes))
    print("Neural Network Precision:", precision_score(y_test, y_pred_classes, average='weighted'))
    print("Neural Network Recall:", recall_score(y_test, y_pred_classes, average='weighted'))
    print("Neural Network F1-score:", f1_score(y_test, y_pred_classes, average='weighted'))
    print("Neural Network AUC:", roc_auc_score(label_binarize(y_test, classes=np.unique(y)), y_pred, multi_class='ovr'))

    # Calculate and print accuracy for each cancer type
    inv_cancer_type_mapping = {v: k for k, v in cancer_type_mapping.items()}
    for label, cancer_type in inv_cancer_type_mapping.items():
        indices = np.where(y_test == label)[0]
        if len(indices) == 0:
            continue
        type_accuracy = accuracy_score(y_test[indices], y_pred_classes[indices])
        print(f"Accuracy for {cancer_type}: {type_accuracy:.4f}")


if __name__ == '__main__':
    # Load and preprocess the dataset
    file_path = '/Users/colt/Downloads/CPSC 571/wetransfer_init_data_design-csv_2024-03-12_2057/normalized_data.csv'
    df = load_large_csv(file_path)
    processed_df, filtered_data_by_cancer_type = preprocess_data(df)

    # Calculate feature importance scores
    fisher_scores = calculate_fisher_ratio(filtered_data_by_cancer_type)
    print("Fisher Scores:", fisher_scores)
    cancer_types = processed_df.index.str.split('_').str[-1]
    cancer_type_mapping = {'BRCA': 0, 'BRACA': 1, 'COCA': 2, 'KICA': 3, 'LECA': 4, 'LUCA': 5}
    y = cancer_types.map(cancer_type_mapping).values
    processed_df.reset_index(drop=True, inplace=True)
    nrs_scores = calculate_nrs_score(processed_df.values, y)
    print("\nNRS Scores:", nrs_scores)

    # Select top features based on combined scores
    feature_names = list(fisher_scores.keys()) + list(processed_df.columns)
    combined_scores = np.concatenate((np.array(list(fisher_scores.values())), nrs_scores))
    feature_scores_df = pd.DataFrame({'Feature': feature_names, 'Combined Score': combined_scores})
    sorted_scores_df = feature_scores_df.sort_values(by='Combined Score', ascending=False)
    k = 1000
    top_features = sorted_scores_df['Feature'][len(fisher_scores):len(fisher_scores)+k]
    X = processed_df[top_features].values
    num_classes = len(np.unique(y))
    input_shape = X.shape[1]

    # Split the data and train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_and_evaluate_model(X_train, X_test, y_train, y_test, cancer_type_mapping)
