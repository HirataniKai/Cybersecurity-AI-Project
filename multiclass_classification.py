# import modules
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import numpy as np
import warnings

def direct_multiclass_train(model_name, X_train, y_train):
    """ 
    This function should take the model_name (“dt”, “knn”, “perceptron”, “nn”) as input
	along with the training data (two Dataframes) and return a trained model.
    """
    if model_name == "dt":
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == "knn":
        model = KNeighborsClassifier(n_neighbors=3)
    elif model_name == "perceptron":
        model = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
    elif model_name == "nn":
        model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

    # fit the model to training data
    model.fit(X_train, y_train)
    
    return model

def direct_multiclass_test(model, X_test, y_test):
    """ 
    This function should take a trained model and evaluate the model on the test data,
	returning an accuracy value.
    """
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    return accuracy

def benign_mal_train(model_name, X_train, y_train):
    """ 
    This function will take the model_name (“dt”, “knn”, “perceptron”, “nn”) as input along
	with the training data (two Dataframes) and return a trained binary model that
	distinguishes between benign and malicious samples. Converts the labels
	to benign and malicious to make it a binary problem.
    """
	# Suppose y_train contains labels ['benign', 'malicious', 'benign', 'benign', 'benign']
	# Convert to binary labels where 'benign' is 0 and 'malicious' is 1. For example the binary_train_binary may now be an array of binary labels [1,1,1,0,0]
    y_train_binary = np.where(y_train == 'benign', 0, 1)
    # Train binary classification model
    if model_name == 'dt':
        model = DecisionTreeClassifier()
    elif model_name == 'knn':
        model = KNeighborsClassifier()
    elif model_name == 'perceptron':
        model = Perceptron()
    elif model_name == 'nn':
        model = MLPClassifier()
    else:
        raise ValueError('Invalid model name')

	# Set feature names
    if hasattr(X_train, 'columns'):  
        feature_names = X_train.columns
        model.feature_names = feature_names
    
    # Suppress X has feature names warning
    warnings.filterwarnings("ignore", category=UserWarning)
    model.fit(X_train, y_train_binary)
    #warnings.filterwarnings("default", category=UserWarning)

    return model

def benign_mal_test(model, X_test):
    """ 
    This function should take a trained model and test data and return a list of predictions
	(one for each test sample). This tests all samples, not just benign ones.
    """
	# Make predictions on test data
    y_pred = model.predict(X_test)
    return y_pred.tolist()

def mal_train(model_name, X_train, y_train):
    """ 
    This function should take the model_name (“dt”, “knn”, “perceptron”, “nn”) as input
	along with the training data (two Dataframes) and return a trained multi-class model that
	distinguishes between different malicious samples. This removes all benign
	samples from the training data before training your model.
    """
    # Remove benign samples from training 
    benign_indices = np.where(y_train == 'benign')[0]
    X_train = np.delete(X_train, benign_indices, axis=0)
    y_train = np.delete(y_train, benign_indices)

    # Train multi-class classification model
    if model_name == 'dt':
        model = DecisionTreeClassifier()
    elif model_name == 'knn':
        model = KNeighborsClassifier()
    elif model_name == 'perceptron':
        model = Perceptron()
    elif model_name == 'nn':
        model = MLPClassifier()

    model.fit(X_train, y_train)
    return model

def mal_test(model, X_test):
    """ 
    This function should take a trained model and test data and return a list of predictions
	(one for each test sample). This tests the model on all test samples, not just malicious ones.
    """
	# Make predictions- testing data
    y_pred = model.predict(X_test)

    return y_pred.tolist()

def evaluate_hierarchical(benign_preds, mal_preds, y_test):
    """ 
    This function should take the list of benign predictions, malicious predictions and the test
	labels as input. The function should return the accuracy of the predictions.
    """
	# Combine the binary and multi-class predictions. Turns into a single set of hierarchical predictions
    hier_preds = [mal_preds[i] if benign_preds[i] == 1 else 0 for i in range(len(benign_preds))]
    
    # Compute the accuracy
    acc = accuracy_score(y_test, hier_preds)
    
    return acc