from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

import numpy as np

def unsup_binary_train(X_train, y_train):
    """ 
    This function should apply K-means with K=2 to the training data and return the trained
    model.
    """
    # Apply KMeans with 2 clusters
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    
    # Fit the KMeans model to the training data
    kmeans.fit(X_train)
    
    # Return the trained KMeans model
    return kmeans
   
def unsup_binary_test(model, X_test, y_test):
    """
    This function should take the trained K-means model and the test data and return the
    accuracy.
    """
    # Predict cluster labels for X_test
    y_pred = model.predict(X_test)

    # Change cluster labels so that it is now binary labels
    y_binary_pred = (y_pred == y_pred.min()).astype(int)

    # Compute accuracy 
    accuracy = accuracy_score(y_test, y_binary_pred)

    return accuracy

def unsup_multiclass_train(X_train, y_train, k):
    """ 
    This function should apply K-means (K=# of different attacks + 1 for benign) to the training
    data and return the trained model.
    """
    # explicitly set n_init to suppress warning
    n_init = 10
    kmeans = KMeans(n_clusters=k, n_init=n_init)

    # Determine the number of clusters (K) to use in K-means. One cluster for each type of attack and one for benign traffic
    n_clusters = k + 1  

    # Fit kmeans to the training data
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
    kmeans.fit(X_train)
    
    return kmeans

def unsup_multiclass_test(model, X_test, y_test):
    """ 
    This function should take the trained K-means model and the test data and return the
    accuracy.
    """
    # predict cluster labels
    y_pred = model.predict(X_test)
   
    # accuracy score is computed by comparing the predicted cluster labels to the true labels
    accuracy = accuracy_score(y_test, y_pred)
    
    # Return the computed accuracy
    return accuracy