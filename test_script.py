import helpers as help
import multiclass_classification as mcc
import feature_selection as fs
import unsupervised_learning as ul
import regression as reg

#Loading Data
df = help.load_data("NetworkIntrusionTrafficData/File1.csv")
df_clean = help.clean_data(df, "remove")

#Split Data
X_train, y_train, X_test, y_test = help.split_data(df_clean, "Label")

#Multi-Class Classification
model = mcc.direct_multiclass_train("dt", X_train, y_train)
acc = mcc.direct_multiclass_test(model, X_test, y_test)
print("MCC Direct Accuracy:", acc)

model = mcc.benign_mal_train("dt", X_train, y_train)
benign_preds = mcc.benign_mal_test(model, X_test)

model = mcc.mal_train("dt", X_train, y_train)
mal_preds = mcc.mal_test(model, X_test)

acc = mcc.evaluate_hierarchical(benign_preds, mal_preds, y_test)
print("MCC Hierarchical Accuracy:", acc)

#Feature Selection
features = fs.find_min_features("dt", X_train, y_train, X_test, y_test, 0.9)
print("Min Features:", features)
features = fs.find_important_features(X_train, y_train)
print("Feature Ordering:", features)

#Unsupervised Learning
model = ul.unsup_binary_train(X_train, y_train)
acc = ul.unsup_binary_test(model, X_test, y_test)
print("Unsupervised Binary Acc:", acc)

model = ul.unsup_multiclass_train(X_train, y_train, 3)
acc = ul.unsup_multiclass_test(model, X_test, y_test)
print("Unsupervised Multi-Class Acc:", acc)

#Regression
#model = reg.benign_regression_train("dt", X_train, y_train)
#thresh = reg.benign_regression_test(model, X_test, y_test)
#acc = reg.benign_regression_evaluate(model, X_test, y_test, thresh)
#print("Regression Threshold:", thresh)
#print("Regression Accuracy:", acc)