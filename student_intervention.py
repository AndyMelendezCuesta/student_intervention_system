# Import libraries
import numpy as np
import pandas as pd
import time 
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.svm import SVC #for Model 1
from sklearn.ensemble import RandomForestClassifier #for Model 2: Randomized Forest
from sklearn.neighbors import KNeighborsClassifier  #for Model 3: K Nearest Neighbors
from sklearn.ensemble import BaggingClassifier #for Model 3: K Nearest Neighbors
 
def load_data():
    """Load the student dataset."""

    student = pd.read_csv("student-data.csv")
    print "Student data read successfully!"
    return student

def explore_student_data(student_data):
    """Calculate the student data statistics"""
    n_students = len(student_data.index)
    n_features = len(student_data.columns)-1
    n_passed = len(student_data[(student_data["passed"]=="yes")])
    n_failed = len(student_data[(student_data["passed"]=="no")])
    grad_rate = float(n_passed) / float(n_passed+n_failed) * 100.0
    feature_cols = list(student_data.columns[:-1])
    target_col = student_data.columns[-1]
    X = student_data[feature_cols]  # feature values for all students\n",
    y = student_data[target_col].replace(['yes', 'no'], [1, 0])  # corresponding targets/labels\n",
    print "Total number of students: " + str(n_students)
    print "Number of students who passed: " + str(n_passed)
    print "Number of students who failed: " + str(n_failed)
    print "Number of features: " + str(n_features)
    print "Graduation rate of the class: " + str(grad_rate)
    print "Feature column(s): " + str(feature_cols)
    print "Target column: " + str(target_col)
    print "Feature values: "
    print X.head()  # print the first 5 rows"
    return X, y

def preprocess_features(student_data):
    """Replace missing data and invalid data"""
    out_Student_data = pd.DataFrame(index=student_data.index)  # output dataframe, initially empty\n",
    # Check each column\n",
    for col, col_data in student_data.iteritems():
        # If data type is non-numeric, try to replace all yes/no values with 1/0\n",
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
            # Note: This should change the data type for yes/no columns to int\n",
            # If still non-numeric, convert to one or more dummy variables\n",
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)  # e.g. 'school' => 'school_GP', 'school_MS'\n",
            out_Student_data = out_Student_data.join(col_data)  # collect column(s) in output dataframe\n",
    return out_Student_data

def strat_shuffle_split(features_data, target_data):
    """Shuffle data to avoid any ordering bias in the dataset"""

    num_size = 95
    sss = StratifiedShuffleSplit(target_data, test_size=num_size, n_iter = 50,random_state=42)

    for train_index, test_index in sss:
       X_train, X_test = features_data.iloc[train_index], features_data.iloc[test_index]
       y_train, y_test = target_data[train_index], target_data[test_index]

    return X_train, y_train, X_test, y_test


def split_data(X, y, num_train):
    """Split data according to num_train"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=num_train)
    return X_train, y_train, X_test, y_test


def train_classifier(clf, X_train, y_train, grid=False):
    print "Training " + clf.__class__.__name__
    start = time.time()
    clf.fit(X_train, y_train)
    if grid:
        clf = clf.best_estimator_
        print "Best estimator: " + str(clf)
    end = time.time()
    training_time = end - start
    print "Done! Training time (secs): " + str(training_time)

# def train_classifier(clf, X_train, y_train, grid):
#     print "Training " + clf.__class__.__name__
#     start = time.time()
#     clf.fit(X_train, y_train)
#     if grid:
#         clf = clf.best_estimator_
#         print "Best estimator: " + str(clf)
#     end = time.time()
#     training_time = end - start
#     print "Done! Training time (secs): " + str(training_time)

# Predict on training set and compute F1 score
def predict_labels(clf, features, target):
    print "Predicting labels using " + str(clf.__class__.__name__)
    start = time.time()
    y_pred = clf.predict(features)
    end = time.time()
    prediction_time = end - start
    print "Done! Prediction time (secs): " + str(prediction_time)
    return f1_score(target.values, y_pred, pos_label=1)

 # Train and predict using different training set sizes
def train_predict(clf, X_train, y_train, X_test, y_test, grid=False):
    print "Training set size: " + str(len(X_train))
    train_classifier(clf, X_train, y_train, grid)
    print "F1 score for training set: " + str(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: " + str(predict_labels(clf, X_test, y_test))
    print "------------------------------------------"

#  # Train and predict using different training set sizes
# def train_predict(clf, X_train, y_train, X_test, y_test, grid):
#     print "Training set size: " + str(len(X_train))
#     train_classifier(clf, X_train, y_train, False)
#     print "F1 score for training set: " + str(predict_labels(clf, X_train, y_train))
#     print "F1 score for test set: " + str(predict_labels(clf, X_test, y_test))
#     print "------------------------------------------"


def create_table(train_num, models, X, y):
    #loop
    for model_name, model in models.items():
        print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
        print "Testing Model " + model_name
        print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
        for size in train_num:
            # Split data
            X_train, y_train, X_test, y_test = split_data(X, y, size)
            #X_train, y_train, X_test, y_test = strat_shuffle_split(features_data, target_data)
            train_predict(model, X_train[:size], y_train[:size], X_test, y_test, False)


def fine_tuning_SVM(parameters, SVM_clf, features_data, target_data):
    # Fine-tuning SVM model\n"
    parameters = {'kernel':('linear','rbf', 'poly','sigmoid'), 'C':[1, 50], 'degree':[3,6]}
    final_svm_clf = GridSearchCV(SVM_clf, parameters, scoring='f1')
    print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    print "Fine-tuning SVM-model: "
    print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    X_train, y_train, X_test, y_test = strat_shuffle_split(features_data, target_data)
    train_predict(final_svm_clf, X_train, y_train, X_test, y_test, grid=True)
    print "Best parameters for the final tuned SVM model is " + str(final_svm_clf.best_params_)


# def fine_tuning_SVM(parameters, SVM_clf, features_data, target_data):
#     # Fine-tuning SVM model\n"
#     parameters = {'kernel':('linear','rbf', 'poly','sigmoid'), 'C':[1, 50], 'degree':[3,6]}
#     final_svm_clf = GridSearchCV(SVM_clf, parameters, scoring='f1')
#     print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
#     print "Fine-tuning SVM-model: "
#     print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
#     X_train, y_train, X_test, y_test = strat_shuffle_split(features_data, target_data)
#     train_predict(final_svm_clf, X_train, y_train, X_test, y_test, True)
#     print "Best parameters for the final tuned SVM model is " + str(final_svm_clf.best_params_)


def main():
    """Analyze the student data. Evaluate and validate the
    performanance of a Decision Tree regressor on the student data.
    Fine tune the model to make prediction on unseen data."""

    # Load data
    student_data = load_data()

    # Explore the data
    X, y = explore_student_data(student_data)

    #Preprocess features
    X = preprocess_features(X) 
    print "Number of preprocessed columns: " + str(len(X.columns)) 
    print "Processed feature columns : " + str(list(X.columns))

    features_data = X
    target_data = y
  
    #Stratified shuffle split 
    X_train, y_train, X_test, y_test = strat_shuffle_split(features_data, target_data)

    print "Training set (X, y): " + str(y_train.shape[0])
    print "Test set (X, y): " + str(y_test.shape[0])    

    #or 
    #print "Training set (X, y): " + str(X_train.shape[0])
    #print "Test set (X, y): " + str(X_test.shape[0])


    #Model 1: Support Vector Classifier Linear Kernel
    #from sklearn.svm import SVC
    SVM_clf = SVC()

    #Model 2: Randomized Forest
    #from sklearn.ensemble import RandomForestClassifier
    RF_clf = RandomForestClassifier(n_estimators=15)

    #Model 3: K Nearest Neighbors
    #from sklearn.neighbors import KNeighborsClassifier
    #from sklearn.ensemble import BaggingClassifier

    bagging_clf = BaggingClassifier(KNeighborsClassifier(n_neighbors=3),max_samples=0.5, max_features=0.5)

    #With training sizes 100, 200, 300
    train_num = [100, 200, 300] 
   
    #models 
    models = {"SVM classifier": SVM_clf, "Randomized Forest": RF_clf, "Bagging Classifier with KNN": bagging_clf}

    #parameters
    parameters = {'kernel':('linear','rbf', 'poly','sigmoid'), 'C':[1, 50], 'degree':[3,6]}

    #create table
    create_table(train_num, models, X, y)

    #Fine-tuning SVM model\n"
    fine_tuning_SVM(parameters, SVM_clf, features_data, target_data)


    print "Finished"


if __name__ == "__main__":
    main()


# --------------------------------- RESULTS -----------------------------------

# Student data read successfully!
# Total number of students: 395
# Number of students who passed: 265
# Number of students who failed: 130
# Number of features: 30
# Graduation rate of the class: 67.0886075949
# Feature column(s): ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']
# Target column: passed
# Feature values: 
#   school sex  age address famsize Pstatus  Medu  Fedu     Mjob      Fjob  \
# 0     GP   F   18       U     GT3       A     4     4  at_home   teacher   
# 1     GP   F   17       U     GT3       T     1     1  at_home     other   
# 2     GP   F   15       U     LE3       T     1     1  at_home     other   
# 3     GP   F   15       U     GT3       T     4     2   health  services   
# 4     GP   F   16       U     GT3       T     3     3    other     other   

#     ...    higher internet  romantic  famrel  freetime goout Dalc Walc health  \
# 0   ...       yes       no        no       4         3     4    1    1      3   
# 1   ...       yes      yes        no       5         3     3    1    1      3   
# 2   ...       yes      yes        no       4         3     2    2    3      3   
# 3   ...       yes      yes       yes       3         2     2    1    1      5   
# 4   ...       yes       no        no       4         3     2    1    2      5   

#   absences  
# 0        6  
# 1        4  
# 2       10  
# 3        2  
# 4        4  

# [5 rows x 30 columns]
# Number of preprocessed columns: 27
# Processed feature columns : ['school_GP', 'school_MS', 'sex_F', 'sex_M', 'address_R', 'address_U', 'famsize_GT3', 'famsize_LE3', 'Pstatus_A', 'Pstatus_T', 'Mjob_at_home', 'Mjob_health', 'Mjob_other', 'Mjob_services', 'Mjob_teacher', 'Fjob_at_home', 'Fjob_health', 'Fjob_other', 'Fjob_services', 'Fjob_teacher', 'reason_course', 'reason_home', 'reason_other', 'reason_reputation', 'guardian_father', 'guardian_mother', 'guardian_other']
# Training set (X, y): 300
# Test set (X, y): 95
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Testing Model SVM classifier
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Training set size: 100
# Training SVC
# Done! Training time (secs): 0.00119495391846
# Predicting labels using SVC
# Done! Prediction time (secs): 0.000564813613892
# F1 score for training set: 0.823529411765
# Predicting labels using SVC
# Done! Prediction time (secs): 0.00131583213806
# F1 score for test set: 0.795918367347
# ------------------------------------------
# Training set size: 200
# Training SVC
# Done! Training time (secs): 0.00307512283325
# Predicting labels using SVC
# Done! Prediction time (secs): 0.00164413452148
# F1 score for training set: 0.823529411765
# Predicting labels using SVC
# Done! Prediction time (secs): 0.0016040802002
# F1 score for test set: 0.78125
# ------------------------------------------
# Training set size: 300
# Training SVC
# Done! Training time (secs): 0.0054190158844
# Predicting labels using SVC
# Done! Prediction time (secs): 0.0032160282135
# F1 score for training set: 0.807157057654
# Predicting labels using SVC
# Done! Prediction time (secs): 0.00113105773926
# F1 score for test set: 0.789808917197
# ------------------------------------------
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Testing Model Randomized Forest
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Training set size: 100
# Training RandomForestClassifier
# Done! Training time (secs): 0.0405828952789
# Predicting labels using RandomForestClassifier
# Done! Prediction time (secs): 0.00231099128723
# F1 score for training set: 0.962406015038
# Predicting labels using RandomForestClassifier
# Done! Prediction time (secs): 0.00241303443909
# F1 score for test set: 0.683291770574
# ------------------------------------------
# Training set size: 200
# Training RandomForestClassifier
# Done! Training time (secs): 0.035796880722
# Predicting labels using RandomForestClassifier
# Done! Prediction time (secs): 0.00321578979492
# F1 score for training set: 0.935714285714
# Predicting labels using RandomForestClassifier
# Done! Prediction time (secs): 0.00213003158569
# F1 score for test set: 0.731884057971
# ------------------------------------------
# Training set size: 300
# Training RandomForestClassifier
# Done! Training time (secs): 0.0431001186371
# Predicting labels using RandomForestClassifier
# Done! Prediction time (secs): 0.00332593917847
# F1 score for training set: 0.922305764411
# Predicting labels using RandomForestClassifier
# Done! Prediction time (secs): 0.00169396400452
# F1 score for test set: 0.740740740741
# ------------------------------------------
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Testing Model Bagging Classifier with KNN
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Training set size: 100
# Training BaggingClassifier
# Done! Training time (secs): 0.0201480388641
# Predicting labels using BaggingClassifier
# Done! Prediction time (secs): 0.00536394119263
# F1 score for training set: 0.820895522388
# Predicting labels using BaggingClassifier
# Done! Prediction time (secs): 0.011342048645
# F1 score for test set: 0.665036674817
# ------------------------------------------
# Training set size: 200
# Training BaggingClassifier
# Done! Training time (secs): 0.0234789848328
# Predicting labels using BaggingClassifier
# Done! Prediction time (secs): 0.00930714607239
# F1 score for training set: 0.81935483871
# Predicting labels using BaggingClassifier
# Done! Prediction time (secs): 0.00933599472046
# F1 score for test set: 0.773162939297
# ------------------------------------------
# Training set size: 300
# Training BaggingClassifier
# Done! Training time (secs): 0.0212380886078
# Predicting labels using BaggingClassifier
# Done! Prediction time (secs): 0.0169909000397
# F1 score for training set: 0.837209302326
# Predicting labels using BaggingClassifier
# Done! Prediction time (secs): 0.00819396972656
# F1 score for test set: 0.773722627737
# ------------------------------------------
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Fine-tuning SVM-model: 
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Training set size: 300
# Training GridSearchCV
# Best estimator: SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
#   decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
#   max_iter=-1, probability=False, random_state=None, shrinking=True,
#   tol=0.001, verbose=False)
# Done! Training time (secs): 0.389611005783
# Predicting labels using GridSearchCV
# Done! Prediction time (secs): 0.00334405899048
# F1 score for training set: 0.802395209581
# Predicting labels using GridSearchCV
# Done! Prediction time (secs): 0.00117683410645
# F1 score for test set: 0.805031446541
# ------------------------------------------
# Best parameters for the final tuned SVM model is {'kernel': 'rbf', 'C': 1, 'degree': 3}
# Finished

# -------------------------------- END -----------------------------------
