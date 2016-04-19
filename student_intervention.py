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

#############################################################
#################### Table of Classifiers ###################
#Source: http://matplotlib.org/examples/pylab_examples/table_demo.html
#############################################################
# """
# Demo of table function to display a table within a plot.
# """
import matplotlib.pyplot as plt
#import numpy as np #already have it

# data = [[  66386,  174296,   75131,  577908,   32015],
#         [  58230,  381139,   78045,   99308,  160454],
#         [  89135,   80552,  152558,  497981,  603535],
#         [  78415,   81858,  150656,  193263,   69638],
#         [ 139361,  331509,  343164,  781380,   52269]]

# columns = ('Freeze', 'Wind', 'Flood', 'Quake', 'Hail')
# rows = ['%d year' % x for x in (100, 50, 20, 10, 5)]

# # Get some pastel shades for the colors
# colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))

# # Reverse colors and text labels to display the last value at the top.
# colors = colors[::-1]

# # Add a table at the bottom of the axes
# the_table = plt.table(cellText=data,
#                       rowLabels=rows,
#                       rowColours=colors,
#                       colLabels=columns,
#                       loc='center')

# #show table
# plt.title('{}'.format("model"))
# plt.axis('off')
# plt.show()
#############################################################
#############################################################
 
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
    #print "Training " + clf.__class__.__name__ ###commented out because of the table func
    start = time.time()
    clf.fit(X_train, y_train)
    if grid:
        clf = clf.best_estimator_
        #print "Best estimator: " + str(clf)  ###commented out because of the table func
    end = time.time()
    training_time = end - start
    #print "Done! Training time (secs): " + str(training_time) ###commented out because of the table func
    return training_time  ###added because of the table func

# Predict on training set and compute F1 score
def predict_labels(clf, features, target):
    #print "Predicting labels using " + str(clf.__class__.__name__) ###commented out because of the table func
    start = time.time()
    y_pred = clf.predict(features)
    end = time.time()
    prediction_time = end - start
    #print "Done! Prediction time (secs): " + str(prediction_time) ###commented out because of the table func
    #return f1_score(target.values, y_pred, pos_label=1) ###commented out because of the table func
    return (prediction_time, f1_score(target.values, y_pred, pos_label=1)) ###modified because of the table func

 # Train and predict using different training set sizes
def train_predict(clf, X_train, y_train, X_test, y_test, grid=False):
    print "------------------------------------------"
    print "Training set size: " + str(len(X_train))
    train_classifier(clf, X_train, y_train, grid)
    print "F1 score for training set: " + str(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: " + str(predict_labels(clf, X_test, y_test))


def create_chart(train_num, models, X, y): #chart
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

def fine_tuning_SVM(parameters, SVM_clf, features_data, target_data, X_train, y_train, X_test, y_test):
    # Fine-tuning SVM model\n"
    final_svm_clf = GridSearchCV(SVM_clf, parameters, scoring='f1')
    print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    print "Fine-tuning SVM-model: "
    print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    #X_train, y_train, X_test, y_test = strat_shuffle_split(features_data, target_data) #removed after review
    train_predict(final_svm_clf, X_train, y_train, X_test, y_test, grid=True)
    print "Best parameters for the final tuned SVM model is " + str(final_svm_clf.best_params_)


##########################################################################################
################################### Creating Table #######################################

def create_table(model, model_name, train_num, X, y): 
    all_data = [] #right location
    columns = ['Training set size: %d' % x for x in train_num]
    rows = [ 
      "Training time of classifier     ", \
      "Prediction time for training set", \
      "F1 score for training set       ", \
      "Prediction time for testing set ", \
      "F1 score for testing set        "]
    for num in train_num:
        data = []
        # Split data
        X_train, y_train, X_test, y_test = split_data(X, y, num)
        #"{0:.2f}".format(round(a,2))
        data = [ \
                "{0:.7f}".format(round(train_classifier(model, X_train, y_train),7)), \
                "{0:.7f}".format(round(predict_labels(model, X_train, y_train)[0],7)), \
                "{0:.7f}".format(round(predict_labels(model, X_train, y_train)[1],7)), \
                "{0:.7f}".format(round(predict_labels(model, X_test, y_test)[0],7)), \
                "{0:.7f}".format(round(predict_labels(model, X_test, y_test)[1],7)) \
                ]
        all_data.append(data) 
    #accomodating data
    all_ordered_data = []
    num_cols = len(all_data)
    num_rows = len(all_data[0])
    #loop
    r_count = 0
    while r_count < num_rows: #loops from 0 up to 4
        ordered_data = []
        c_count = 0
        while c_count < num_cols: #visits all_data[0], all_data[1], all_data[2]
            ordered_data.append(all_data[c_count][r_count])
            c_count += 1
        all_ordered_data.append(ordered_data)
        r_count += 1

    #Get some pastel shades for the colors
    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
    # Reverse colors and text labels to display the last value at the top.
    colors = colors[::-1]

    #Add a table at the bottom of the axes
    the_table = plt.table(cellText=all_ordered_data,
                          rowLabels=rows,    ##row labels must be length 3 
                          rowColours=colors,
                          colLabels=columns,
                          loc='center')
    #show table
    plt.title('{}'.format(model_name))
    plt.axis('off')
    plt.savefig("table_{}.png".format(model_name)) #k components, where k is clusters
    plt.show()


def all_tables(models, train_num, X, y):
    #loop
    for model_name, model in models.items():
        print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
        print "Testing Model " + model_name
        print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
        create_table(model, model_name, train_num, X, y)

#################################################################################################
#################################################################################################

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

    #Model 4: Bagging with K Nearest Neighbors
    #from sklearn.neighbors import KNeighborsClassifier
    #from sklearn.ensemble import BaggingClassifier

    bagging_clf = BaggingClassifier(KNeighborsClassifier(n_neighbors=3),max_samples=0.5, max_features=0.5)

    #With training sizes 100, 200, 300
    train_num = [100, 200, 300] 
   
    #models 
    models = {"SVM classifier": SVM_clf, "Randomized Forest": RF_clf, "Bagging Classifier with KNN": bagging_clf}

    #parameters
    parameters = {'kernel':('linear','rbf', 'poly','sigmoid'), 'C':[1, 50], 'degree':[3,6]}

    #creates CHARTS :P #################################################################
    create_chart(train_num, models, X, y)

    #fine_tuning_SVM(parameters, SVM_clf, features_data, target_data)#original code
    fine_tuning_SVM(parameters, SVM_clf, features_data, target_data, X_train, y_train, X_test, y_test)#modified code after review

    #tuning RF_clf model
    #RF_clf = RandomForestClassifier(n_estimators=15)
    #RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)

    #tuning bagging_clf
    #BaggingClassifier(base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)
    #BaggingClassifier(KNeighborsClassifier(n_neighbors=3),max_samples=0.5, max_features=0.5)

    all_tables(models, train_num, X, y)

    print "Finished"


if __name__ == "__main__":
    main()

#After REVIEW, considering creating a class of SVM and include: 
#      features_data, target_data, X_train, y_train, X_test, y_test #as instance attributes
#Perhaps that would be the ideal if we had beyond 10 parameters
#https://jeffknupp.com/blog/2014/06/18/improve-your-python-python-classes-and-object-oriented-programming/

# --------------------------------- RESULTS (1st run)-----------------------------------
# Andreas-MacBook-Pro-2:student_intervention andreamelendezcuesta$ python student_intervention_Edited.py
# Student data read successfully!
# Total number of students: 395
# Number of students who passed: 265
# Number of students who failed: 130
# Number of features: 30
# Graduation rate of the class: 67.09%
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
# Number of preprocessed columns: 48
# Processed feature columns : ['school_GP', 'school_MS', 'sex_F', 'sex_M', 'age', 'address_R', 'address_U', 'famsize_GT3', 'famsize_LE3', 'Pstatus_A', 'Pstatus_T', 'Medu', 'Fedu', 'Mjob_at_home', 'Mjob_health', 'Mjob_other', 'Mjob_services', 'Mjob_teacher', 'Fjob_at_home', 'Fjob_health', 'Fjob_other', 'Fjob_services', 'Fjob_teacher', 'reason_course', 'reason_home', 'reason_other', 'reason_reputation', 'guardian_father', 'guardian_mother', 'guardian_other', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']
# Training set (X, y): 300
# Test set (X, y): 95
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Testing Model SVM classifier
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------------------------------------
# Training set size: 100
# F1 score for training set: (0.0014960765838623047, 0.90140845070422537)
# F1 score for test set: (0.0036630630493164062, 0.80266075388026614)
# ------------------------------------------
# Training set size: 200
# F1 score for training set: (0.004441976547241211, 0.87248322147651014)
# F1 score for test set: (0.003981828689575195, 0.83439490445859865)
# ------------------------------------------
# Training set size: 300
# F1 score for training set: (0.009087085723876953, 0.85224839400428265)
# F1 score for test set: (0.0032799243927001953, 0.83544303797468344)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Testing Model Randomized Forest
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------------------------------------
# Training set size: 100
# F1 score for training set: (0.0027239322662353516, 0.99346405228758172)
# F1 score for test set: (0.004172801971435547, 0.7604395604395604)
# ------------------------------------------
# Training set size: 200
# F1 score for training set: (0.0028870105743408203, 1.0)
# F1 score for test set: (0.0031049251556396484, 0.80405405405405417)
# ------------------------------------------
# Training set size: 300
# F1 score for training set: (0.003303050994873047, 0.99516908212560384)
# F1 score for test set: (0.0026388168334960938, 0.76335877862595414)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Testing Model Bagging Classifier with KNN
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------------------------------------
# Training set size: 100
# F1 score for training set: (0.03318309783935547, 0.86956521739130432)
# F1 score for test set: (0.016726016998291016, 0.77729257641921379)
# ------------------------------------------
# Training set size: 200
# F1 score for training set: (0.01868915557861328, 0.83934426229508186)
# F1 score for test set: (0.017992019653320312, 0.82781456953642396)
# ------------------------------------------
# Training set size: 300
# F1 score for training set: (0.03153514862060547, 0.85779816513761475)
# F1 score for test set: (0.012219905853271484, 0.88461538461538469)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Fine-tuning SVM-model: 
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------------------------------------
# Training set size: 300
# F1 score for training set: (0.005442142486572266, 0.80239520958083843)
# F1 score for test set: (0.0019931793212890625, 0.80503144654088055)
# Best parameters for the final tuned SVM model is {'kernel': 'sigmoid', 'C': 1, 'degree': 3}
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Testing Model SVM classifier
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Testing Model Randomized Forest
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Testing Model Bagging Classifier with KNN
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Finished

# --------------------------------- RESULTS (2nd run)-----------------------------------

# Andreas-MacBook-Pro-2:student_intervention andreamelendezcuesta$ python student_intervention_Edited.py
# Student data read successfully!
# Total number of students: 395
# Number of students who passed: 265
# Number of students who failed: 130
# Number of features: 30
# Graduation rate of the class: 67.09%
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
# Number of preprocessed columns: 48
# Processed feature columns : ['school_GP', 'school_MS', 'sex_F', 'sex_M', 'age', 'address_R', 'address_U', 'famsize_GT3', 'famsize_LE3', 'Pstatus_A', 'Pstatus_T', 'Medu', 'Fedu', 'Mjob_at_home', 'Mjob_health', 'Mjob_other', 'Mjob_services', 'Mjob_teacher', 'Fjob_at_home', 'Fjob_health', 'Fjob_other', 'Fjob_services', 'Fjob_teacher', 'reason_course', 'reason_home', 'reason_other', 'reason_reputation', 'guardian_father', 'guardian_mother', 'guardian_other', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']
# Training set (X, y): 300
# Test set (X, y): 95
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Testing Model SVM classifier
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------------------------------------
# Training set size: 100
# F1 score for training set: (0.0013489723205566406, 0.83870967741935476)
# F1 score for test set: (0.003253936767578125, 0.81390593047034765)
# ------------------------------------------
# Training set size: 200
# F1 score for training set: (0.0036728382110595703, 0.86850152905198774)
# F1 score for test set: (0.0036530494689941406, 0.76129032258064511)
# ------------------------------------------
# Training set size: 300
# F1 score for training set: (0.007561206817626953, 0.86521739130434783)
# F1 score for test set: (0.0026350021362304688, 0.79999999999999993)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Testing Model Randomized Forest
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------------------------------------
# Training set size: 100
# F1 score for training set: (0.002549886703491211, 1.0)
# F1 score for test set: (0.003406047821044922, 0.77130044843049317)
# ------------------------------------------
# Training set size: 200
# F1 score for training set: (0.002914905548095703, 1.0)
# F1 score for test set: (0.002868175506591797, 0.75444839857651247)
# ------------------------------------------
# Training set size: 300
# F1 score for training set: (0.0034399032592773438, 0.99754299754299758)
# F1 score for test set: (0.002666950225830078, 0.78518518518518521)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Testing Model Bagging Classifier with KNN
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------------------------------------
# Training set size: 100
# F1 score for training set: (0.01047515869140625, 0.86842105263157887)
# F1 score for test set: (0.016479015350341797, 0.78260869565217384)
# ------------------------------------------
# Training set size: 200
# F1 score for training set: (0.018923044204711914, 0.86585365853658536)
# F1 score for test set: (0.018131017684936523, 0.79470198675496684)
# ------------------------------------------
# Training set size: 300
# F1 score for training set: (0.028747081756591797, 0.8642533936651583)
# F1 score for test set: (0.015403985977172852, 0.86274509803921562)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Fine-tuning SVM-model: 
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------------------------------------
# Training set size: 300
# F1 score for training set: (0.005850076675415039, 0.80239520958083843)
# F1 score for test set: (0.0021622180938720703, 0.80503144654088055)
# Best parameters for the final tuned SVM model is {'kernel': 'sigmoid', 'C': 1, 'degree': 3}
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Testing Model SVM classifier
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Testing Model Randomized Forest
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Testing Model Bagging Classifier with KNN
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Finished


# --------------------------------- RESULTS (3rd run)-----------------------------------

# Andreas-MacBook-Pro-2:student_intervention andreamelendezcuesta$ python student_intervention_Edited.py
# Student data read successfully!
# Total number of students: 395
# Number of students who passed: 265
# Number of students who failed: 130
# Number of features: 30
# Graduation rate of the class: 67.09%
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
# Number of preprocessed columns: 48
# Processed feature columns : ['school_GP', 'school_MS', 'sex_F', 'sex_M', 'age', 'address_R', 'address_U', 'famsize_GT3', 'famsize_LE3', 'Pstatus_A', 'Pstatus_T', 'Medu', 'Fedu', 'Mjob_at_home', 'Mjob_health', 'Mjob_other', 'Mjob_services', 'Mjob_teacher', 'Fjob_at_home', 'Fjob_health', 'Fjob_other', 'Fjob_services', 'Fjob_teacher', 'reason_course', 'reason_home', 'reason_other', 'reason_reputation', 'guardian_father', 'guardian_mother', 'guardian_other', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']
# Training set (X, y): 300
# Test set (X, y): 95
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Testing Model SVM classifier
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------------------------------------
# Training set size: 100
# F1 score for training set: (0.0012679100036621094, 0.89610389610389607)
# F1 score for test set: (0.0031549930572509766, 0.80168776371308004)
# ------------------------------------------
# Training set size: 200
# F1 score for training set: (0.0037801265716552734, 0.86468646864686471)
# F1 score for test set: (0.003793954849243164, 0.82315112540192925)
# ------------------------------------------
# Training set size: 300
# F1 score for training set: (0.007536172866821289, 0.86382978723404258)
# F1 score for test set: (0.002640962600708008, 0.80519480519480524)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Testing Model Randomized Forest
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------------------------------------
# Training set size: 100
# F1 score for training set: (0.0027899742126464844, 1.0)
# F1 score for test set: (0.003164052963256836, 0.77876106194690253)
# ------------------------------------------
# Training set size: 200
# F1 score for training set: (0.0029480457305908203, 0.99630996309963105)
# F1 score for test set: (0.002919912338256836, 0.78200692041522479)
# ------------------------------------------
# Training set size: 300
# F1 score for training set: (0.0033371448516845703, 0.99473684210526314)
# F1 score for test set: (0.0029230117797851562, 0.86792452830188682)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Testing Model Bagging Classifier with KNN
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------------------------------------
# Training set size: 100
# F1 score for training set: (0.010005950927734375, 0.86111111111111116)
# F1 score for test set: (0.017061948776245117, 0.76252723311546844)
# ------------------------------------------
# Training set size: 200
# F1 score for training set: (0.01871013641357422, 0.87234042553191493)
# F1 score for test set: (0.018385887145996094, 0.81720430107526887)
# ------------------------------------------
# Training set size: 300
# F1 score for training set: (0.03169894218444824, 0.85462555066079293)
# F1 score for test set: (0.015124082565307617, 0.78321678321678312)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Fine-tuning SVM-model: 
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------------------------------------
# Training set size: 300
# F1 score for training set: (0.005799055099487305, 0.80239520958083843)
# F1 score for test set: (0.0020859241485595703, 0.80503144654088055)
# Best parameters for the final tuned SVM model is {'kernel': 'sigmoid', 'C': 1, 'degree': 3}
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Testing Model SVM classifier
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Testing Model Randomized Forest
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Testing Model Bagging Classifier with KNN
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Finished

# --------------------------------------- END ------------------------------------------






