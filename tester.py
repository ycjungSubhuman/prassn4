import numpy as np

def _read_data():
    all_data = np.loadtxt('data/iris.data', delimiter=',', usecols=(0,1,2,3))
    assert (len(all_data) == 150)
    return [all_data[0:50]/10, all_data[50:100]/10, all_data[100:150]/10]

def _split_train_test(data, iteration, num_chunk):
    chunks = np.split(data, num_chunk)
    train_set = np.array([chunks[i] for i in range(0, num_chunk) if i != iteration])
    return (train_set.reshape((num_chunk - 1)*(len(data)//num_chunk), data.shape[1]), chunks[iteration])

class IrisClassifierTester:
    '''
    Init tester

    Args:
    folds : cross validation folds
    '''
    def __init__(self, folds=5):
        self.classified_data_list = _read_data();
        self.folds = folds
        self.train_test_pairs = [[] for _ in range(0, folds)]
        self.num_class = len(self.classified_data_list)
        self.confusions = np.array([np.zeros((self.num_class, self.num_class)) for i in range(0, self.folds)])
        for i in range(0, folds):
            self.train_test_pairs[i] = [_split_train_test(d, i, folds) for d in self.classified_data_list]

    def run(self):
        for i in range(0, self.folds):
            classifier = self._get_classifier([train_set for (train_set, _) in self.train_test_pairs[i]])
            test_set = [(self.train_test_pairs[i][j][1], j) for j in range(0, self.num_class)]

            for (rows, gt) in test_set:
                for d in rows:
                    class_of_data = classifier.classify(d)
                    self.confusions[i][class_of_data][gt] += 1

    def _get_classifier(self, train_set):
        raise 'Not Implemented'

    def get_confusion_matrix(self):
        return self.confusion

    def get_precision(self):
        return sum([np.trace(self.confusions[i]) for i in range(0, self.folds)]) / (self.folds * np.sum(self.confusions[0]))

