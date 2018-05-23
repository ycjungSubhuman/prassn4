from tester import *
from classifier import *

kernels = ['poly', 'rbf', 'sigmoid']
input_modes = ['raw', 'pca', 'lda']

for input_mode in input_modes:
    print ("*******************************************INPUT : {}".format(input_mode))
    print ("######## ONE VS ONE ########")

    for k in kernels:
        print('mode-kernel: {}'.format(k))
        t = OneVsOneTester(k, input_mode=input_mode)
        t.run()
        print(t.get_confusion_matrix())
        print(t.get_precision())


    print ("######## ONE VS ALL ########")

    for k in kernels:
        print('mode-kernel: {}'.format(k))
        t = OneVsAllTester(k, input_mode=input_mode)
        t.run()
        print(t.get_confusion_matrix())
        print(t.get_precision())

    print ("######## LIBSVM-STYLE ########")

    for k in kernels:
        print('mode-kernel: {}'.format(k))
        t = LibSvmTester(k, input_mode=input_mode)
        t.run()
        print(t.get_confusion_matrix())
        print(t.get_precision())

    print ("######## 3 ENSEMBLE - ONE VS ONE ########")

    for k in kernels:
        print('mode-kernel: {}'.format(k))
        t = EnsembleSvmTester(kernel=k, svm_constructor=OneVsOneSvmClassifier, ensemble_size=3, input_mode=input_mode)
        t.run()
        print(t.get_confusion_matrix())
        print(t.get_precision())

    print ("######## 3 ENSEMBLE - ONE VS ALL ########")

    for k in kernels:
        print('mode-kernel: {}'.format(k))
        t = EnsembleSvmTester(kernel=k, svm_constructor=OneVsAllSvmClassifier, ensemble_size=3, input_mode=input_mode)
        t.run()
        print(t.get_confusion_matrix())
        print(t.get_precision())

    print ("######## 3 ENSEMBLE - LIBSVM ########")

    for k in kernels:
        print('mode-kernel: {}'.format(k))
        t = EnsembleSvmTester(kernel=k, svm_constructor=LibSvmClassifier, ensemble_size=3, input_mode=input_mode)
        t.run()
        print(t.get_confusion_matrix())
        print(t.get_precision())


    print ("######## 5 ENSEMBLE - ONE VS ONE ########")

    for k in kernels:
        print('mode-kernel: {}'.format(k))
        t = EnsembleSvmTester(kernel=k, svm_constructor=OneVsOneSvmClassifier, ensemble_size=5, input_mode=input_mode)
        t.run()
        print(t.get_confusion_matrix())
        print(t.get_precision())

    print ("######## 5 ENSEMBLE - ONE VS ALL ########")

    for k in kernels:
        print('mode-kernel: {}'.format(k))
        t = EnsembleSvmTester(kernel=k, svm_constructor=OneVsAllSvmClassifier, ensemble_size=5, input_mode=input_mode)
        t.run()
        print(t.get_confusion_matrix())
        print(t.get_precision())

    print ("######## 5 ENSEMBLE - LIBSVM ########")

    for k in kernels:
        print('mode-kernel: {}'.format(k))
        t = EnsembleSvmTester(kernel=k, svm_constructor=LibSvmClassifier, ensemble_size=5, input_mode=input_mode)
        t.run()
        print(t.get_confusion_matrix())
        print(t.get_precision())
