from tester import *
from classifier import *

kernels = ['poly', 'rbf', 'sigmoid']

print ("######## ONE VS ONE ########")

for k in kernels:
    print('mode-kernel: {}'.format(k))
    t = OneVsOneTester(k)
    t.run()
    print(t.get_confusion_matrix())
    print(t.get_precision())


print ("######## ONE VS ALL ########")

for k in kernels:
    print('mode-kernel: {}'.format(k))
    t = OneVsAllTester(k)
    t.run()
    print(t.get_confusion_matrix())
    print(t.get_precision())

print ("######## LIBSVM-STYLE ########")

for k in kernels:
    print('mode-kernel: {}'.format(k))
    t = LibSvmTester(k)
    t.run()
    print(t.get_confusion_matrix())
    print(t.get_precision())

print ("######## 3 ENSEMBLE - ONE VS ONE ########")

for k in kernels:
    print('mode-kernel: {}'.format(k))
    t = EnsembleSvmTester(kernel=k, svm_constructor=OneVsOneSvmClassifier, ensemble_size=3)
    t.run()
    print(t.get_confusion_matrix())
    print(t.get_precision())

print ("######## 3 ENSEMBLE - ONE VS ALL ########")

for k in kernels:
    print('mode-kernel: {}'.format(k))
    t = EnsembleSvmTester(kernel=k, svm_constructor=OneVsAllSvmClassifier, ensemble_size=3)
    t.run()
    print(t.get_confusion_matrix())
    print(t.get_precision())

print ("######## 3 ENSEMBLE - LIBSVM ########")

for k in kernels:
    print('mode-kernel: {}'.format(k))
    t = EnsembleSvmTester(kernel=k, svm_constructor=LibSvmClassifier, ensemble_size=3)
    t.run()
    print(t.get_confusion_matrix())
    print(t.get_precision())


print ("######## 5 ENSEMBLE - ONE VS ONE ########")

for k in kernels:
    print('mode-kernel: {}'.format(k))
    t = EnsembleSvmTester(kernel=k, svm_constructor=OneVsOneSvmClassifier, ensemble_size=5)
    t.run()
    print(t.get_confusion_matrix())
    print(t.get_precision())

print ("######## 5 ENSEMBLE - ONE VS ALL ########")

for k in kernels:
    print('mode-kernel: {}'.format(k))
    t = EnsembleSvmTester(kernel=k, svm_constructor=OneVsAllSvmClassifier, ensemble_size=5)
    t.run()
    print(t.get_confusion_matrix())
    print(t.get_precision())

print ("######## 5 ENSEMBLE - LIBSVM ########")

for k in kernels:
    print('mode-kernel: {}'.format(k))
    t = EnsembleSvmTester(kernel=k, svm_constructor=LibSvmClassifier, ensemble_size=5)
    t.run()
    print(t.get_confusion_matrix())
    print(t.get_precision())
