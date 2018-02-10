# EECS 445 - Winter 2018
# Project 1 - project1.py

import pandas as pd
import numpy as np
import itertools
import string

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from helper import *


def select_classifier(penalty='l2', c=1.0, degree=1, r=0.0, class_weight='balanced'):
    """
    Return a linear svm classifier based on the given
    penalty function and regularization parameter c.
    """
    if penalty == 'l1':
        return LinearSVC(penalty='l1', dual=False, C=c, class_weight=class_weight)
    elif degree == 1:
        return SVC(kernel="linear", C=c, degree=1, class_weight=class_weight)

    # TODO: Optionally implement this helper function if you would like to
    # instantiate your SVM classifiers in a single function. You will need
    # to use the above parameters throughout the assignment.


def extract_dictionary(df):
    """
    Reads a panda dataframe, and returns a dictionary of distinct words
    mapping from each distinct word to its index (ordered by when it was found).
    Input:
        df: dataframe/output of load_data()
    Returns:
        a dictionary of distinct words that maps each distinct word
        to a unique index corresponding to when it was first found while
        iterating over all words in each review in the dataframe df
    """
    word_dict = {}
    for i, s in df['content'].iteritems():
        for p in string.punctuation:
            s = s.replace(p, " ")
        s = s.lower()
        li = s.split()
        for w in li:
            if w in word_dict:
                continue;
            size = word_dict.__len__()
            word_dict[w] = size



    # TODO: Implement this function
    return word_dict


def generate_feature_matrix(df, word_dict):
    """
    Reads a dataframe and the dictionary of unique words
    to generate a matrix of {1, 0} feature vectors for each review.
    Use the word_dict to find the correct index to set to 1 for each place
    in the feature vector. The resulting feature matrix should be of
    dimension (number of reviews, number of words).
    Input:
        df: dataframe that has the ratings and labels
        word_list: dictionary of words mapping to indices
    Returns:
        a feature matrix of dimension (number of reviews, number of words)
    """
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    for i, review in df['content'].iteritems():
        for p in string.punctuation:
            review = review.replace(p, " ")
        review = review.lower()
        li = review.split()
        for w in li:
            if w in word_dict:
                feature_matrix[i][word_dict[w]] = 1

    # TODO: Implement this function
    return feature_matrix


def cv_performance(clf, X, y, k=5, metric="accuracy"):
    """
    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates the k-fold cross-validation performance metric for classifier
    clf by averaging the performance across folds.
    Input:
        clf: an instance of SVC()
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
    Returns:
        average 'test' performance across the k folds as np.float64
    """
    # TODO: Implement this function
    #HINT: You may find the StratifiedKFold from sklearn.model_selection
    #to be useful
    scores = []
    skf = StratifiedKFold(n_splits=k)
    n = y.shape


    #Put the performance of the model on each fold in the scores array
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        if metric == "auroc":
            y_score = clf.decision_function(X_test)
            scores.append(metrics.roc_auc_score(y_test, y_score))
        y_pred = clf.predict(X_test)
        if metric == "accuracy":
            scores.append(metrics.accuracy_score(y_test, y_pred))
        if metric == "precision":
            scores.append(metrics.precision_score(y_test, y_pred))
        elif metric == "sensitivity":
            tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
            scores.append(np.float64(tp) / (tp + fn))
        elif metric == "specificity":
            tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
            scores.append(np.float64(tn) / (tn + fp))
        elif metric == "f1-score":
            scores.append(metrics.f1_score(y_test, y_pred))



    #And return the average performance across all fold splits.
    return np.array(scores).mean()


def select_param_linear(X, y, k=5, metric="accuracy", C_range = [], penalty='l2'):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy',
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
    Returns:
        The parameter value for a linear-kernel SVM that maximizes the
        average 5-fold CV performance.
    """
    best_c = C_range[0]
    clf = SVC(kernel="linear", C=C_range[0], class_weight="balanced")
    best_performance = cv_performance(clf, X, y, k, metric)
    for c in C_range:
        clf = SVC(kernel="linear",  C=c, class_weight="balanced")
        perf = cv_performance(clf, X, y, k, metric)
        print("C: ", c, ", perf: ", perf)
        if perf > best_performance:
            best_c = c
            best_performance = perf
    print("Metric: ", metric, ", Best_C: ", best_c, ", performance: ", best_performance)
    return best_c


    # TODO: Implement this function
    #HINT: You should be using your cv_performance function here
    #to evaluate the performance of each SVM


def plot_weight(X,y,penalty,metric,C_range):
    """
    Takes as input the training data X and labels y and plots the L0-norm
    (number of nonzero elements) of the coefficients learned by a classifier
    as a function of the C-values of the classifier.
    """

    print("Plotting the number of nonzero entries of the parameter vector as a function of C")
    norm0 = []

    # TODO: Implement this part of the function
    #Here, for each value of c in C_range, you should
    #append to norm0 the L0-norm of the theta vector that is learned
    #when fitting an L2-penalty, degree=1 SVM to the data (X, y)
    for c in C_range:
        clf = select_classifier(penalty=penalty, c=c)
        clf.fit(X, y)
        norm0.append(np.count_nonzero(clf.coef_))
        print ('C: ', c, ', norm0: ', norm0)


    #This code will plot your L0-norm as a function of c
    plt.plot(C_range, norm0)
    plt.xscale('log')
    plt.legend(['L0-norm'])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")
    plt.title('Norm-'+penalty+'_penalty.png')
    plt.savefig('Norm-'+penalty+'_penalty.png')
    plt.close()


def select_param_quadratic(X, y, k=5, metric="accuracy", param_range=[]):
    """
        Sweeps different settings for the hyperparameters of an quadratic-kernel SVM,
        calculating the k-fold CV performance for each setting on X, y.
        Input:
            X: (n,d) array of feature vectors, where n is the number of examples
               and d is the number of features
            y: (n,) array of binary labels {1,-1}
            k: an int specifying the number of folds (default=5)
            metric: string specifying the performance metric (default='accuracy'
                     other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                     and 'specificity')
            parameter_values: a (num_param, 2)-sized array containing the
                parameter values to search over. The first column should
                represent the values for C, and the second column should
                represent the values for r. Each row of this array thus
                represents a pair of parameters to be tried together.
        Returns:
            The parameter value(s) for a quadratic-kernel SVM that maximize
            the average 5-fold CV performance
    """
    # TODO: Implement this function

    best_c = param_range[0][0]
    best_r = param_range[0][1]
    clf = SVC(kernel="poly", degree=2, C=best_c, coef0=best_r, class_weight="balanced")
    best_performance = cv_performance(clf, X, y, k, metric)
    for p in param_range:
        clf = SVC(kernel="poly", degree=2, C=p[0], coef0=p[1], class_weight="balanced")
        perf = cv_performance(clf, X, y, k, metric)
        print("P: ", p, ", perf: ", perf)
        if perf > best_performance:
            best_c = p[0]
            best_r = p[1]
            best_performance = perf
    print("Metric: ", metric, ", C: ", best_c, ", r: ", best_r, " performance: ", best_performance)
    return best_c, best_r
    



    # Hint: This will be very similar to select_param_linear, except
    # the type of SVM model you are using will be different...


def performance(y_true, y_pred=[], metric="accuracy", y_score=[]):
    """
    Calculates the performance metric as evaluated on the true labels
    y_true versus the predicted labels y_pred.
    Input:
        y_true: (n,) array containing known labels
        y_pred: (n,) array containing predicted scores
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
    Returns:
        the performance as an np.float64
    """
    # TODO: Implement this function



    #Put the performance of the model on each fold in the scores array


    if metric == "auroc":
        return metrics.roc_auc_score(y_true, y_score)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    if metric == "accuracy":
        return np.float64(tn + tp) / len(y_pred)
    elif metric == "precision":
        return np.float64(tp) / (tp + fp)
    elif metric == "sensitivity":
        return np.float64(tp) / (tp + fn)
    elif metric == "specificity":
        return np.float64(tn) / (tn + fp)
    elif metric == "f1-score":
        prec = np.float64(tp)/ (tp + fp)
        sensi = np.float64(tp)/ (tp + fn)
        return (2 * prec * sensi)/(prec + sensi)

        # This is an optional but very useful function to implement.
        # See the sklearn.metrics documentation for pointers on how to implement
        # the requested metrics.


def q_two(X_train,dictionary_binary):
    print("2")
    print(dictionary_binary.__len__())
    n, d = np.shape(X_train)
    average_non_zero = np.sum(X_train)/ np.float64(n)
    print(average_non_zero)


def q_three_one_c(X_train,Y_train):
    print("3.1c")
    best_c_accuracy = select_param_linear(X_train, Y_train, k=5, metric="accuracy", C_range=[1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3])
    best_c_f1 = select_param_linear(X_train, Y_train, k=5, metric="f1-score", C_range=[1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3])
    best_c_auroc = select_param_linear(X_train, Y_train, k=5, metric="auroc", C_range=[1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3])
    best_c_preci = select_param_linear(X_train, Y_train, k=5, metric="precision", C_range=[1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3])
    best_c_sensi = select_param_linear(X_train, Y_train, k=5, metric="sensitivity", C_range=[1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3])
    best_c_speci = select_param_linear(X_train, Y_train, k=5, metric="specificity", C_range=[1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3])


def q_3_1d(X_train, Y_train, Y_test, X_test):
    print("3.1d")
    clf = SVC(kernel="linear", C=0.01, class_weight="balanced")
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    print("accuracy: ", performance(Y_test, y_pred, metric="accuracy", y_score=[]))
    print("f1: ", performance(Y_test, y_pred, metric="f1-score", y_score=[]))

    y_Score = clf.decision_function(X_test)
    print("auroc: ", performance(Y_test, y_pred=[], metric="auroc", y_score=y_Score))

    print("precision: ", performance(Y_test, y_pred, metric="precision", y_score=[]))
    print("sensitivity: ", performance(Y_test, y_pred, metric="sensitivity", y_score=[]))
    print("specificity: ", performance(Y_test, y_pred, metric="specificity", y_score=[]))


def q_3_1f(X_train, Y_train, dictionary_binary):
    print("3.1f")
    clf = SVC(kernel="linear", C=0.1, class_weight="balanced")
    clf.fit(X_train, Y_train)
    theta = clf.coef_
    # clf.coef_ return a matrix of 1 x n!!!!
    index = np.argsort(theta)[0]
    for i in range(4):
        pos = index[i]
        for w, ind in dictionary_binary.items():
            if ind == pos:
                print(w, ": ", theta[0][pos])

    for i in range(4):
        pos = index[len(index) - i - 1]
        for w, ind in dictionary_binary.items():
            if ind == pos:
                print(w, ": ", theta[0][pos])


def q_3_2(X_train, Y_train, X_test, Y_test):
    print("3.2")
    Range = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    param_range_G = []
    for i in Range:
        for j in Range:
            param_range_G.append([i, j])

    g_c, g_r = select_param_quadratic(X_train, Y_train, k=5, metric='auroc',param_range=param_range_G)
    clf = SVC(kernel="poly", degree=2, C=g_c, coef0=g_r, class_weight="balanced")
    clf.fit(X_train, Y_train)
    y_score = clf.decision_function(X_test)
    print("testing Y AUROC: ", metrics.roc_auc_score(Y_test, y_score))

    range_c = np.random.uniform(-3, 3, 25)
    range_r = np.random.uniform(-3, 3, 25)
    param_range_R = []
    for i in range(25):
        param_range_R.append([10 ** range_c[i], 10 ** range_r[i]])

    r_c, r_r = select_param_quadratic(X_train, Y_train, k=5, metric='auroc',param_range=param_range_R)

    clf = SVC(kernel="poly", degree=2, C=r_c, coef0=r_r, class_weight="balanced")
    clf.fit(X_train, Y_train)
    y_score = clf.decision_function(X_test)
    print("testing Y AUROC: ", metrics.roc_auc_score(Y_test, y_score))

def q_3_4a(X_train, Y_train):
    print("3.4a")
    C_range = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    best_c = C_range[0]
    clf = LinearSVC(penalty='l1', dual=False, C=C_range[0], class_weight="balanced")
    best_performance = cv_performance(clf, X_train, Y_train, k=5, metric='auroc')
    for c in C_range:
        clf = LinearSVC(penalty='l1', dual=False, C=c, class_weight="balanced")
        perf = cv_performance(clf, X_train, Y_train, k=5, metric='auroc')
        print("C:  ", c, ", perf: ", perf)
        if perf > best_performance:
            best_c = c
            best_performance = perf
    print("C: ", best_c, ", performance: ", best_performance)
    return best_c


def q_4_1(X_train, X_test, Y_train, Y_test):
    print("4.1")
    clf = select_classifier(penalty='l2', c=0.01, class_weight={-1: 10, 1: 1})
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    print("accuracy: ", performance(Y_test, y_pred, metric="accuracy", y_score=[]))
    print("f1: ", performance(Y_test, y_pred, metric="f1-score", y_score=[]))

    y_Score = clf.decision_function(X_test)
    print("auroc: ", performance(Y_test, y_pred, metric="auroc", y_score=y_Score))
    print("precision: ", performance(Y_test, y_pred, metric="precision", y_score=[]))
    print("sensitivity: ", performance(Y_test, y_pred, metric="sensitivity", y_score=[]))
    print("specificity: ", performance(Y_test, y_pred, metric="specificity", y_score=[]))

def q_4_2(IMB_features, IMB_labels, IMB_test_features, IMB_test_labels):
    print("4.2")
    clf = select_classifier(penalty='l2', c=0.01, class_weight={-1: 1, 1: 1})
    clf.fit(IMB_features, IMB_labels)
    y_pred = clf.predict(IMB_test_features)
    print("accuracy: ", performance(IMB_test_labels, y_pred, metric="accuracy", y_score=[]))
    print("f1: ", performance(IMB_test_labels, y_pred, metric="f1-score", y_score=[]))

    y_Score = clf.decision_function(IMB_test_features)
    print("auroc: ", performance(IMB_test_labels, y_pred, metric="auroc", y_score=y_Score))
    print("precision: ", performance(IMB_test_labels, y_pred, metric="precision", y_score=[]))
    print("sensitivity: ", performance(IMB_test_labels, y_pred, metric="sensitivity", y_score=[]))
    print("specificity: ", performance(IMB_test_labels, y_pred, metric="specificity", y_score=[]))


def q_4_3(IMB_features, IMB_labels, IMB_test_features, IMB_test_labels):
    print("4.3")
    print("class wait: -1:4, 1: 1")
    clf = select_classifier(penalty='l2', c=0.01, class_weight={-1: 4, 1: 1})
    clf.fit(IMB_features, IMB_labels)
    y_pred = clf.predict(IMB_test_features)
    print("accuracy: ", performance(IMB_test_labels, y_pred, metric="accuracy", y_score=[]))
    print("f1: ", performance(IMB_test_labels, y_pred, metric="f1-score", y_score=[]))

    y_Score = clf.decision_function(IMB_test_features)
    print("auroc: ", performance(IMB_test_labels, y_pred, metric="auroc", y_score=y_Score))
    print("precision: ", performance(IMB_test_labels, y_pred, metric="precision", y_score=[]))
    print("sensitivity: ", performance(IMB_test_labels, y_pred, metric="sensitivity", y_score=[]))
    print("specificity: ", performance(IMB_test_labels, y_pred, metric="specificity", y_score=[]))


def q_4_4a1(IMB_features, IMB_labels, IMB_test_features, IMB_test_labels):
    print("custom setting")
    clf = select_classifier(penalty='l2', c=0.01, class_weight={-1: 4, 1: 1})
    clf.fit(IMB_features, IMB_labels)
    y_Score = clf.decision_function(IMB_test_features)
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(IMB_test_labels, y_Score)
    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
             label='4:1 AUC = %0.2f' % roc_auc)

    clf = select_classifier(penalty='l2', c=0.01, class_weight={-1: 1, 1: 1})
    clf.fit(IMB_features, IMB_labels)
    y_Score = clf.decision_function(IMB_test_features)
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(IMB_test_labels, y_Score)
    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
    plt.plot(false_positive_rate, true_positive_rate, 'g',
             label='1:1 AUC = %0.2f' % roc_auc)

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def select_param_linear_multi(X, y, k=5, metric="accuracy", C_range = [], penalty='l2'):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy',
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
    Returns:
        The parameter value for a linear-kernel SVM that maximizes the
        average 5-fold CV performance.
    """
    best_c = C_range[0]
    clf = OneVsRestClassifier(LinearSVC(penalty='l1', dual=False, C=C_range[0], class_weight='balanced'))
    best_performance = cv_performance(clf, X, y, k, metric)
    for c in C_range:
        clf = OneVsRestClassifier(LinearSVC(penalty='l1', dual=False, C=c, class_weight='balanced'))
        perf = cv_performance(clf, X, y, k, metric)
        if perf > best_performance:
            best_c = c
            best_performance = perf
    print("C: ", best_c, ", performance: ", best_performance)
    return best_c

def extract_dictionary_5(df):
    word_dict = {}
    better_dict  = {}
    best_dict = {}
    for i, s in df['content'].iteritems():
        for p in string.punctuation:
            s = s.replace(p, " ")
        s = s.lower()
        li = s.split()
        for w in li:
            if w not in word_dict:
                size = word_dict.__len__()
                word_dict[w] = size
            elif w not in better_dict:
                better_dict[w] = better_dict.__len__()
            elif w not in best_dict:
                best_dict[w] = best_dict.__len__()
    print(best_dict)
    return best_dict




def q_5(multiclass_features, multiclass_labels, multiclass_dictionary, heldout_features):
    print("5")
    print("OneVSRest")
    #C_range = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    rand = np.random.uniform(-1, 1, 10)
    C_range = []
    for i in range(10):
        C_range.append(10 ** rand[i])
    best_c = C_range[0]
    clf = OneVsRestClassifier(LinearSVC(penalty='l1', dual=False, C=C_range[0], class_weight='balanced'))
    best_performance = cv_performance(clf, multiclass_features, multiclass_labels, k=5, metric='accuracy')
    for c in C_range:
        clf = OneVsRestClassifier(LinearSVC(penalty='l1', dual=False, C=c, class_weight='balanced'))
        perf = cv_performance(clf, multiclass_features, multiclass_labels, k=5, metric='accuracy')
        print("C: ", c, ", perf: ", perf)
        if perf > best_performance:
            best_c = c
            best_performance = perf
    print("Best_C: ", best_c, ", performance: ", best_performance)
    clf = OneVsRestClassifier(LinearSVC(penalty='l1', dual=False, C=best_c, class_weight='balanced'))
    clf.fit(multiclass_features, multiclass_labels)
    y_pred = clf.predict(heldout_features)
    generate_challenge_labels(y_pred, "hengma")

    print("OneVSOne")
    best_c = C_range[0]
    clf = OneVsOneClassifier(LinearSVC(penalty='l1', dual=False, C=C_range[0], class_weight='balanced'))
    best_performance = cv_performance(clf, multiclass_features, multiclass_labels, k=5, metric='accuracy')
    for c in C_range:
        clf = OneVsOneClassifier(LinearSVC(penalty='l1', dual=False, C=c, class_weight='balanced'))
        perf = cv_performance(clf, multiclass_features, multiclass_labels, k=5, metric='accuracy')
        print("C: ", c, ", perf: ", perf)
        if perf > best_performance:
            best_c = c
            best_performance = perf
    print("Best_C: ", best_c, ", performance: ", best_performance)
    clf = OneVsOneClassifier(LinearSVC(penalty='l1', dual=False, C=best_c, class_weight='balanced'))
    clf.fit(multiclass_features, multiclass_labels)
    y_pred = clf.predict(heldout_features)




def main():
    # Read binary data
    # NOTE: READING IN THE DATA WILL NOT WORK UNTIL YOU HAVE FINISHED
    #       IMPLEMENTING generate_feature_matrix AND extract_dictionary
    X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data()
    IMB_features, IMB_labels = get_imbalanced_data(dictionary_binary)
    IMB_test_features, IMB_test_labels = get_imbalanced_test(dictionary_binary)
    '''
    # TODO: Questions 2, 3, 4
    # 2
    q_two(X_train, dictionary_binary)

    # 3.1c
    q_three_one_c(X_train, Y_train)

    #3.1d
    q_3_1d(X_train, Y_train, Y_test, X_test)

    #3.1e
    print("3.1e")
    plot_weight(X_train, Y_train, "l2", "accuracy", [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3])
    #3.1f
    q_3_1f(X_train, Y_train, dictionary_binary)

    #3.2
    q_3_2(X_train,Y_train, X_test, Y_test)

    #3.4.a
    q_3_4a(X_train, Y_train)
    #3.4.b
    print("3.4b")
    plot_weight(X_train, Y_train, "l1", "accuracy", C_range=[1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3])

    #4.1
    q_4_1(X_train, X_test, Y_train, Y_test)
    #4.2
    q_4_2(IMB_features, IMB_labels, IMB_test_features, IMB_test_labels)
    #4.3
    q_4_3(IMB_features, IMB_labels, IMB_test_features, IMB_test_labels)

    #4.4
    q_4_4a1(IMB_features, IMB_labels, IMB_test_features, IMB_test_labels)
    '''

    # Read multiclass data
    # TODO: Question 5: Apply a classifier to heldout features, and then use
    #       generate_challenge_labels to print the predicted labels
    multiclass_features, multiclass_labels, multiclass_dictionary = get_multiclass_training_data()
    heldout_features = get_heldout_reviews(multiclass_dictionary)
    #5
    q_5(multiclass_features, multiclass_labels, multiclass_dictionary, heldout_features)

if __name__ == '__main__':
    main()
