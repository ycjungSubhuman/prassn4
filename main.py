from tester import *

kernels = ['poly', 'rbf', 'sigmoid']

print ("######## ONE VS ONE ########")

for k in kernels:
    print('mode-kernel: {}'.format(k))
    t1 = OneVsOneTester(k)
    t1.run()
    print(t1.get_confusion_matrix())
    print(t1.get_precision())


print ("######## ONE VS ALL ########")

for k in kernels:
    print('mode-kernel: {}'.format(k))
    t2 = OneVsAllTester(k)
    t2.run()
    print(t2.get_confusion_matrix())
    print(t2.get_precision())

print ("######## LIBSVM-STYLE ########")

for k in kernels:
    print('mode-kernel: {}'.format(k))
    t3 = LibSvmTester(k)
    t3.run()
    print(t3.get_confusion_matrix())
    print(t3.get_precision())
