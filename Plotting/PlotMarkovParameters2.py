"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 20
Date: November 2021
Python: 3.7.7
"""


import matplotlib.pyplot as plt
import numpy.linalg as LA


def plotMarkovParameters2(list1, list2, name1, name2, num):

    len1 = len(list1)
    len2 = len(list2)

    list1 = list1[0:min(len1, len2)]
    list2 = list2[0:min(len1, len2)]

    plt.figure(num=num, figsize=[12, 12])
    mp1 = []
    mp2 = []
    diff = []
    for e in list1:
        mp1.append(LA.norm(e))
    i = 0
    for e in list2:
        mp2.append(LA.norm(e))
        diff.append(mp1[i] - LA.norm(e))
        i+=1
    plt.subplot(2, 1, 1)
    plt.plot(mp1)
    plt.plot(mp2)
    plt.legend([name1, name2])
    plt.xlabel('Time steps')
    plt.ylabel('Frobenius norm of Markov Parameters')
    plt.subplot(2, 1, 2)
    plt.plot(diff)
    print(diff)
    plt.legend('Error')
    plt.xlabel('Time steps')
    plt.ylabel('Frobenius norm of Error Markov Parameters')

    plt.show()
