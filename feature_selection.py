from itertools import combinations
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

# Trains a regression model on the input data and returns the trained model.
def train_regression_model(model_name):
    if model_name == 'dt':
        return DecisionTreeClassifier()
    elif model_name == 'knn':
        return KNeighborsClassifier()
    elif model_name == 'perceptron':
        return Perceptron()
    elif model_name == 'nn':
        return MLPClassifier()
    else:
        raise ValueError('Invalid model_name: {}'.format(model_name))

def find_min_features(model_name, X_train, y_train, X_test, y_test, accuracy):
    """ 
    This function should take a model_name, training data, test data and a target accuracy as
    input and return a list of feature names that produce the desired accuracy. Accuracy will
    be a real-valued number from 0 to 1.0.
    """
    # Get the list of feature names
    feature_names = X_train.columns.tolist()
    
    # Iterate possible combos of features
    for i in range(1, len(feature_names) + 1):
        for combo in combinations(feature_names, i):
            # Train current combination of features
            clf = train_regression_model(model_name)
            clf.fit(X_train[list(combo)], y_train)
            # Evaluate the model (testset)
            acc = clf.score(X_test[list(combo)], y_test)
            # If the accuracy is >= to the target accuracy, return the feature names
            if acc >= accuracy:
                return list(combo)
    
    # If not, return an empty list
    return []
     
    
def find_important_features(X_train, y_train):
    """ 
    This function should take the training data as input and return a list of feature names
    ranked from most important to least important. 
    """
    # The code trains a decision tree classifier model on the training data X_train with corresponding labels y_train
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    # Retrieve and arrange the importance scores of features in a descending order
    feature_importances = list(zip(X_train.columns, clf.feature_importances_))
    feature_importances.sort(key=lambda x: x[1], reverse=True)
    
    # Retrieve the feature names and output them as a list
    important_features = [f[0] for f in feature_importances]
    return important_features

