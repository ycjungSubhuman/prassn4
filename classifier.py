import numpy as np
from sklearn import svm as sksvm
import svm as libsvm


'''
General classifier interface
'''
class Classifier:

    '''
    Train a N-class classifier
    
    Args:
    train_set_list : a list of numpy array of shape (<number of data in a class>, <dimension of one data>)
    '''
    def __init__(train_feature, train_label):
        raise 'Not Implemented'

    '''
    Classify given data into groups
    
    Args:
    data : a numpy array of shape (<number of data>, <dimension of one data>)
    
    Return:
    a numby array of shape (<number of data>, )
    '''
    def classify(data):
        raise 'Not Implemented'


class OneVsOneSvmClassifier(Classifier):

    def __init__(train_feature, train_label):
        pass

    def classify(data):
        pass

class OneVsAllSvmClassifier(Classifier):

    def __init__(train_feature, train_label):
        pass

    def classify(data):
        pass


class LibSvmClassifier(Classifier):

    def __init__(train_feature, train_label):
        pass

    def classify(data):
        pass
