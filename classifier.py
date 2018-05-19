import numpy as np
from sklearn import svm as sksvm
import svm as libsvm
from itertools import combinations

'''
Split training dataset in one-vs-one fashion
'''
def _prepare_one_vs_one_pair(classified_data_list):
    num_class = len(classified_data_list)
    labels = combinations([i for i in range(num_class)])
    return [(np.vstack((X1,X2)), (np.append(np.zeros(X1.shape[0]), np.ones(X2.shape[1]))))
            for (X1, X2) in combinations(classified_data_list, 2)]
'''
Split training dataset in one-vs-all fashion
'''
def _prepare_one_vs_all_pair(classified_data_list):
    num_class = len(classified_data_list)
    labels = combinations([i for i in range(num_class)])
    return [(np.vstack((X1,X2)), (np.append(np.zeros(X1.shape[0]), np.ones(X2.shape[1]))))
            for (X1, X2) in combinations(classified_data_list, 2)]

'''
General classifier interface
'''

class Classifier:

    '''
    Train a N-class classifier

    Args: 
    classified_data_list : a list of numpy array of shape (<number of data in the class>, <data dimension>)
    kernel : kernel type. Must be one of ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed')
    '''
    def __init__(classified_data_list, kernel):
        raise 'Not Implemented'

    '''
    Classify given data into groups
    
    Args:
    data : a numpy array of shape (<number of data>, <dimension of one data>)
    
    Return:
    a numpy array of shape (<number of data>, )
    '''
    def classify(data):
        raise 'Not Implemented'


class OneVsOneSvmClassifier(Classifier):

    '''
    Make One VS One SVM classifier
    '''
    def __init__(classified_data_list, kernel):
        self.kernel = kernel
        self._svm = sksvm.SVC(kernel=kernel)
        self._svm.fit(train_feature, train_label)

    def classify(data):
        pass

class OneVsAllSvmClassifier(Classifier):

    '''
    Make One VS All SVM classifier
    '''
    def __init__(classified_data_list, kernel):
        pass

    def classify(data):
        pass


class LibSvmClassifier(Classifier):

    '''
    Make libsvm classifier

    Construct a multi-class classifier using libsvm
    '''
    def __init__(classified_data_list, kernel):
        pass

    def classify(data):
        pass
