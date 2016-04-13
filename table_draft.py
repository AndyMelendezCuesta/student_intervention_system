from astropy.table import Table, Column
import numpy as np
# #http://docs.astropy.org/en/stable/table/construct_table.html
# t = Table()
# t['a'] = [1, 4]
# t['b'] = Column([2.0, 5.0], unit='cm', description='Velocity')
# t['c'] = ['x', 'y']

# t = Table(names=('a', 'b', 'c'), dtype=('f4', 'i4', 'S2'))
# t.add_row((1, 2.0, 'x'))
# t.add_row((4, 5.0, 'y'))

# t = Table(dtype=[('a', 'f4'), ('b', 'i4'), ('c', 'S2')])

def new_table(train_num, model, X, y):
    t = Table()
    all_columns= []
    main_col = Column([ "Training time of classifier     ", \
                        "Prediction time for training set", "F1 score for training set       ", \
                        "Prediction time for testing set ", "F1 score for testing set        "])
    all_columns.append(main_col)
    #then:
    for num in train_num:
        # Split data
        X_train, y_train, X_test, y_test = split_data(X, y, num)
        #X_train, y_train, X_test, y_test = strat_shuffle_split(features_data, target_data)
        t['{}'.format(num)] = Column([str(train_classifier(clf, X_train, y_train, grid)), \
                str(predict_labels(clf, X_train, y_train)[0]), str(predict_labels(clf, X_train, y_train)[1]), \
                str(predict_labels(clf, X_test, )[0]), str(predict_labels(clf, X_test, y_test)[1])])
        all_columns.append(t['{}'.format(num)]) #all_cols = [main_cols, t[100], t[200], t[300]]
    #table
    t = Table(all_cols, names=('Training set size               ', '{}'.format(train_num[0]), \
                               '{}'.format(train_num[1]), '{}'.format(train_num[2]))
   return t #would it be necessary to put "print t" ?

#how to call new_table func
#ummm

#With training sizes 100, 200, 300
train_num = [100, 200, 300] 

#models 
models = {"SVM classifier": SVM_clf, "Randomized Forest": RF_clf, "Bagging Classifier with KNN": bagging_clf}

def all_tables(models, train_num, X, y)
    #loop
    for model_name, model in models.items():
        print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
        print "Testing Model " + model_name
        print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
        #create table
        new_table(train_num, model, X, y)
    print "-----------------------------------------------"



# print "Training set size               "                        str(len(X_train))
# #                       Bagging Classifier with KNN
# print "Training time of {}".format(clf.__class__.__name__)      str(train_classifier(clf, X_train, y_train, grid))
# print "Prediction time for training set"                        str(predict_labels(clf, X_train, y_train)[0])
# print "F1 score for training set       "                        str(predict_labels(clf, X_train, y_train)[1])
# print "Prediction time for testing set "                        str(predict_labels(clf, X_test, y_test)[0])
# print "F1 score for testing set        "                        str(predict_labels(clf, X_test, y_test)[1])
