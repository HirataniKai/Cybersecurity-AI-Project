Kai Hiratani
CS 345
5-14-23

Final Project Overview:
This project revolves around the analysis of network traffic data using the Intrusion Detection Evaluation Dataset (https://www.unb.ca/cic/datasets/ids-2017.html). 
The dataset, which is provided in a zip file on Webcampus, consists of six files that will be used for training and testing.
We have 5 different .py files that will be utilized in the given test_script. 
This includes: helpers.py, multiclass_classification.py, feature_selection.py, unsupervised_learning.py, and regression.py.

helpers.py:
This code has three functions- load, clean, and split a dataset. The load_data function reads a file and stores the data in a Pandas Dataframe. 
The clean_data function removes NaN values and non-numerical features. The split_data function splits the data into training and testing sets. 
This separates the target variable from the features. In the end, it'll return four DataFrames: X_train, y_train, X_test, and y_test, with an 80:20 ratio for training and testing.

multiclass_classification.py:
The code has six functions that import classifiers and metrics. It trains and tests binary and multiclass classification models to distinguish between benign and malicious samples. 
The "evaluate_hierarchical" function combines binary and multiclass predictions to make hierarchical predictions with accuracy.

feature_selection.py:
This code includes functions to train and evaluate regression models using different algorithms. 
Includes a function that determines the minimum number of features needed to reach a specific level of accuracy, and another function that ranks the significance of features.

unsupervised_learning:
These functions use K-means for training and testing models in binary and multiclass classification.
The functions use K-means with 2 clusters. The multiclass functions use K-means with the number of clusters equal to the total number of attack types plus 1 for benign traffic.
The test functions provide accuracy values for the predictions.