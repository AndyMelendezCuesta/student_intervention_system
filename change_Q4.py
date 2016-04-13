#For converting the .ipynb file to .pdf: ipython nbconvert --to pdf student_intervention.ipynb  
#or                                      jupyter nbconvert --to pdf student_intervention.ipynb
# Import libraries
import numpy as np
import pandas as pd
import time 
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.svm import SVC #for Model 1: SVC 
from sklearn.ensemble import RandomForestClassifier #for Model 2: Randomized Forest
from sklearn.neighbors import KNeighborsClassifier  #for Model 3: Bagging Classifier with KNN
from sklearn.ensemble import BaggingClassifier #for Model 3: Bagging Classifier with KNN
 
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
    print "Total number of students: {}".format(n_students)
    print "Number of students who passed: {}".format(n_passed)
    print "Number of students who failed: {}".format(n_failed)
    print "Number of features: {}".format(n_features)
    print "Graduation rate of the class: {:.2f}%".format(grad_rate)
    feature_cols = list(student_data.columns[:-1])
    target_col = student_data.columns[-1]
    X = student_data[feature_cols]  # feature values for all students\n",
    y = student_data[target_col].replace(['yes', 'no'], [1, 0])  # corresponding targets/labels\n",
    print "Feature column(s): {}".format(feature_cols)
    print "Target column: {}".format(target_col)
    print "Feature values: "
    print X.head()  # print the first 5 rows
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
    print "------------------------------------------"
    print "Training set size: " + str(len(X_train))
    train_classifier(clf, X_train, y_train, grid)
    print "F1 score for training set: " + str(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: " + str(predict_labels(clf, X_test, y_test))

#  # Train and predict using different training set sizes
# def train_predict(clf, X_train, y_train, X_test, y_test, grid):
#     print "------------------------------------------"
#     print "Training set size: " + str(len(X_train))
#     train_classifier(clf, X_train, y_train, False)
#     print "F1 score for training set: " + str(predict_labels(clf, X_train, y_train))
#     print "F1 score for test set: " + str(predict_labels(clf, X_test, y_test))



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

#After REVIEW, considering creating a class of SVM and include: 
#      features_data, target_data, X_train, y_train, X_test, y_test #as instance attributes
#https://jeffknupp.com/blog/2014/06/18/improve-your-python-python-classes-and-object-oriented-programming/

def fine_tuning_SVM(parameters, SVM_clf, features_data, target_data, X_train, y_train, X_test, y_test):
    # Fine-tuning SVM model\n"
    final_svm_clf = GridSearchCV(SVM_clf, parameters, scoring='f1')
    print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    print "Fine-tuning SVM-model: "
    print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    #X_train, y_train, X_test, y_test = strat_shuffle_split(features_data, target_data) #removed after review
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

    #Model 2: KNeighborsClassifier
    #KN_clf = KNeighborsClassifier()


    #Model 3: Randomized Forest
    #from sklearn.ensemble import RandomForestClassifier
    RF_clf = RandomForestClassifier(n_estimators=15)

    #Model 4: K Nearest Neighbors
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

    #Fine-tuning SVM model
    #fine_tuning_SVM(parameters, SVM_clf, features_data, target_data)#original code
    fine_tuning_SVM(parameters, SVM_clf, features_data, target_data, X_train, y_train, X_test, y_test)#modified code after review

    #tuning RF_clf model
    #RF_clf = RandomForestClassifier(n_estimators=15)
    #RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)

    #tuning bagging_clf
    #BaggingClassifier(base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)
    #BaggingClassifier(KNeighborsClassifier(n_neighbors=3),max_samples=0.5, max_features=0.5)


    print "Finished"


if __name__ == "__main__":
    main()

