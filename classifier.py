import sys
import numpy as np
import svmutil
from sklearn import svm as sksvm
from itertools import combinations
from collections import defaultdict, Counter

import util

# Penalty parameter C
C = 60.0
MAX_ITER = 500

'''
Split training dataset in one-vs-one fashion
'''
def _prepare_one_vs_one_pair(classified_data_list):
    num_class = len(classified_data_list)
    names = combinations([i for i in range(num_class)], 2)
    return ([(np.vstack((X1,X2)), (np.append(np.zeros(X1.shape[0]), np.ones(X2.shape[0]))))
             for (X1, X2) in combinations(classified_data_list, 2)], names)

def _mk_one_vs_all_data(classified_data_list, ind):
    one = np.empty([0, classified_data_list[0].shape[1]])
    rest = np.empty([0, classified_data_list[0].shape[1]])
    for i in range(len(classified_data_list)):
        if i == ind:
            one = np.vstack((one, classified_data_list[i]))
        else:
            rest = np.vstack((rest, classified_data_list[i]))
    alldata = np.vstack((one, rest))
    labels = np.append(np.zeros(one.shape[0]), np.ones(rest.shape[0]))

    return (alldata, labels)

'''
Split training dataset in one-vs-all fashion
'''
def _prepare_one_vs_all_pair(classified_data_list):
    num_class = len(classified_data_list)
    names = [i for i in range(num_class)]
    result = ([_mk_one_vs_all_data(classified_data_list, i) for i in range(len(classified_data_list))], names)
    return result

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
    def __init__(self, classified_data_list, kernel):
        raise 'Not Implemented'

    '''
    Classify given data into groups
    
    Args:
    data : a numpy array of shape (<dimension of one data>, )
    
    Return:
    a numpy array of shape (<number of data>, )
    '''
    def classify(self, data):
        raise 'Not Implemented'


class OneVsOneSvmClassifier(Classifier):

    '''
    Make One VS One SVM classifier
    '''
    def __init__(self, classified_data_list, kernel):
        self.kernel = kernel
        self._svms = {}
        (pairs, names) = _prepare_one_vs_one_pair(classified_data_list)
        for ((data, labels), name) in zip(pairs, names):
            svm = sksvm.SVC(kernel=kernel, C=C, max_iter=MAX_ITER)
            svm.fit(data, labels)
            self._svms[name] = svm

    def classify(self, data):
        data = data.reshape([1, data.shape[0]])
        # Majority Voting
        votes = [defaultdict(lambda: 0) for i in range(data.shape[0])]
        for ((zero_label, one_label), svm) in self._svms.items():
            pred_labels = svm.predict(data)
            for i, l in enumerate(pred_labels):
                if l == 0 :
                    votes[i][zero_label] += 1
                else:
                    votes[i][one_label] += 1

        # Find the majority
        label_result = np.zeros(data.shape[0])
        for i in range(data.shape[0]):
            vote_max = 0
            verdict = -1
            for (label, num_vote) in votes[i].items():
                if num_vote >= vote_max:
                    verdict = label
                    vote_max = num_vote
            assert(verdict != -1)
            label_result[i] = verdict

        return int(label_result[0])
                    

class OneVsAllSvmClassifier(Classifier):

    '''
    Make One VS All SVM classifier
    '''
    def __init__(self, classified_data_list, kernel):
        self.kernel = kernel
        self._svms = {}
        (pairs, names) = _prepare_one_vs_all_pair(classified_data_list)
        for ((data, labels), name) in zip(pairs, names):
            svm = sksvm.SVC(kernel=kernel, C=C, max_iter=MAX_ITER)
            svm.fit(data, labels)
            self._svms[name] = svm

    def classify(self, data):
        data = data.reshape([1, data.shape[0]])

        # Farthest distance
        dists = {}
        for (one_label, svm) in self._svms.items():
            pred_dists = svm.decision_function(data)
            dists[one_label] = pred_dists[0]

        dist_min = sys.maxsize
        verdict = -1
        for (i, d) in dists.items():
            if dist_min >= d:
                verdict = i
                dist_min = d

        assert(verdict != -1)

        return verdict


class LibSvmClassifier(Classifier):

    '''
    Make libsvm classifier

    Construct a multi-class classifier using libsvm
    '''
    def __init__(self, classified_data_list, kernel):
        self.kernel = kernel
        data = np.empty([0, classified_data_list[0].shape[1]])
        labels = np.empty([0])
        for i, d in enumerate(classified_data_list):
            data = np.vstack((data, d))
            labels = np.append(labels, i*np.ones(d.shape[0]))
        problem = svmutil.svm_problem(labels.tolist(), data.tolist())
        kmap = {'poly': 1, 'rbf': 2, 'sigmoid': 3}
        param = svmutil.svm_parameter('-q -c {} -t {}'.format(C, kmap[kernel]))
        self.svm = svmutil.svm_train(problem, param)

    def classify(self, data):
        data = data.reshape([1, data.shape[0]])
        labels, _, _ = svmutil.svm_predict([0]*data.shape[0], data.tolist(), self.svm, '-q')
        return int(labels[0])

class EnsembleSvmClassifier(Classifier):
    def __init__(self, classified_data_list, kernel, svm_constructor=LibSvmClassifier, ensemble_size=3):
        self._classifiers = []

        for _ in range(ensemble_size):
            data = classified_data_list.copy()
            util.shuffle(data)
            util.take_n(data, 40)
            self._classifiers.append(svm_constructor(data, kernel))

    def classify(self, data):
        return Counter([c.classify(data) for c in self._classifiers]).most_common()[0][0]

