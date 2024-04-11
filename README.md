
## CPSC 571 Group G-6 ‘Classifying Cancer Subtypes from Gene Expression Data’

## Authors
- Desmarais, Jesse (00292117)
- Nguyen, Colton (30080293)
- Roszell, Darren (30163669)

## Requirements

The following Python packages are required to run the models:
- python
- pandas
- numpy
- scikit-learn
- tensorflow
- ReliefF

### Installation
`pip install -r requirements.txt`

The following models are implemented in this repository:

1. **Fisher Ratio + K-Folds**
2. **Fisher Ratio + Neighbourhood Rough Set (NRS) + K-Folds**
3. **Fisher Ratio + Neighbourhood Rough Set (NRS) + Keras Tuner (Neural Network Only)**
4. **Neighbourhood Rough Set (NRS) + K-Folds**
5. **No Feature Selection + K-Folds**
6. **Principal Component Analysis (PCA) + K-Folds**
7. **ReliefF + K-Folds**

Each model is implemented in a separate Python script within the `Models` folder.

### Fisher Ratio + K-Folds

This model uses the Fisher ratio as a feature selection method, followed by K-fold cross-validation to evaluate the performance of various machine learning algorithms.

**Parameters:**
- `n_mgsRFtree`: Number of trees in a Random Forest during Multi Grain Scanning.
- `window`: List of window sizes to use during Multi Grain Scanning.
- `stride`: Step used when slicing the data.
- `cascade_test_size`: Split fraction or absolute number for cascade training set splitting.
- `n_cascadeRF`: Number of Random Forests in a cascade layer.
- `n_cascadeRFtree`: Number of trees in a single Random Forest in a cascade layer.
- `min_samples_mgs`: Minimum number of samples in a node to perform a split during the training of Multi-Grain Scanning Random Forest.
- `min_samples_cascade`: Minimum number of samples in a node to perform a split during the training of Cascade Random Forest.
- `cascade_layer`: Maximum number of cascade layers allowed.
- `tolerance`: Accuracy tolerance for the casacade growth.
- `n_jobs`: The number of jobs to run in parallel for any Random Forest fit and predict.

### Fisher Ratio + Neighbourhood Rough Set (NRS) + K-Folds

This model combines the Fisher ratio and Neighbourhood Rough Set (NRS) as feature selection methods, followed by K-fold cross-validation to evaluate the performance of various machine learning algorithms.

**Parameters:**
- Same as the "Fisher Ratio + K-Folds" model.

### Fisher Ratio + Neighbourhood Rough Set (NRS) + Keras Tuner (Neural Network Only)

This model combines the Fisher ratio and Neighbourhood Rough Set (NRS) as feature selection methods, and uses Keras Tuner to optimize the hyperparameters of a neural network model.

**Parameters:**
- `shape_1X`: Shape of a single sample element [n_lines, n_cols].
- `n_mgsRFtree`: Number of trees in a Random Forest during Multi Grain Scanning.
- `window`: List of window sizes to use during Multi Grain Scanning.
- `stride`: Step used when slicing the data.
- `cascade_test_size`: Split fraction or absolute number for cascade training set splitting.
- `n_cascadeRF`: Number of Random Forests in a cascade layer.
- `n_cascadeRFtree`: Number of trees in a single Random Forest in a cascade layer.
- `min_samples_mgs`: Minimum number of samples in a node to perform a split during the training of Multi-Grain Scanning Random Forest.
- `min_samples_cascade`: Minimum number of samples in a node to perform a split during the training of Cascade Random Forest.
- `cascade_layer`: Maximum number of cascade layers allowed.
- `tolerance`: Accuracy tolerance for the casacade growth.
- `n_jobs`: The number of jobs to run in parallel for any Random Forest fit and predict.

### Neighbourhood Rough Set (NRS) + K-Folds

This model uses the Neighbourhood Rough Set (NRS) as a feature selection method, followed by K-fold cross-validation to evaluate the performance of various machine learning algorithms.

**Parameters:**
- Same as the "Fisher Ratio + K-Folds" model.

### No Feature Selection + K-Folds

This model does not perform any feature selection and directly applies K-fold cross-validation to evaluate the performance of various machine learning algorithms.

**Parameters:**
- Same as the "Fisher Ratio + K-Folds" model.

### Principal Component Analysis (PCA) + K-Folds

This model uses Principal Component Analysis (PCA) as a dimensionality reduction technique, followed by K-fold cross-validation to evaluate the performance of various machine learning algorithms.

**Parameters:**
- `n_components`: The number of principal components to keep.

### ReliefF + K-Folds

This model uses the ReliefF algorithm as a feature selection method, followed by K-fold cross-validation to evaluate the performance of various machine learning algorithms.

**Parameters:**
- `n_features_to_select`: The number of features to select using ReliefF.
- `n_neighbors`: The number of neighbors to consider in the ReliefF algorithm.

## Usage

To run the models, simply execute the corresponding Python script in the `Models` folder. The scripts will load the data, preprocess it, apply the feature selection method (if applicable), and perform the K-fold cross-validation.

The results of the model evaluations, including accuracy, precision, recall, F1-score, and AUC, will be printed to the console.
