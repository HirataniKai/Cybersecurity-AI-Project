from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder

import numpy as np

""""
This function should take the model name, and the training data as input and return a
regression model for benign data.
"""
def benign_regression_train(model_name, X_train, y_train):
    if model_name == "dt":
        model = DecisionTreeRegressor(random_state=42)
    elif model_name == "knn":
        model = KNeighborsRegressor(n_neighbors=3)
    elif model_name == "perceptron":
        model = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
    elif model_name == "nn":
        model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

    # preprocess the target variable using label encoding
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    # fit the model to the training data
    model.fit(X_train, y_train)
    
    return model
    
"""
This function should take the model and test data as input. Use the test data to identify a
threshold distance that would correctly classify the malicious data as malicious. Return
this threshold.
"""
def benign_regression_test(model, X_test, y_test):
    # predict the target variable for the test data
    y_pred = model.predict(X_test)

    # calculate the mean absolute error between the predicted and actual target variable
    mae = np.mean(abs(y_pred - y_test))

    # set the threshold distance as the mean absolute error
    threshold = mae

    return threshold

"""  
This function should take the model, test data and threshold as input and return the
accuracy for binary malicious traffic identification.
"""
def benign_regression_evaluate(model, X_test, y_test, threshold):
    # predict the target variable for the test data
    y_pred = model.predict(X_test)

    # classify the test data as benign or malicious based on the threshold distance
    y_pred_binary = np.where(abs(y_pred - y_test) <= threshold, 0, 1)

    # calculate the accuracy of the classification
    accuracy = np.mean(y_pred_binary == 1)

    return accuracy
    
    

