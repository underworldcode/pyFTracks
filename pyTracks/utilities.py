import numpy as np
from scipy import stats
import random
import sys


def drawbinom(I, prob):
    # Look at scipy.stats.binom...option binom.rvs
    """Random draw from binomial distribution

    Utility function:
    Draw from a binomial distribution:
    Only return if the draw is different than 0
    """
    Ns = 0
    while Ns == 0:
        A = np.random.RandomState()
        Ns = A.binomial(I, prob)
    return Ns


def create_distribution(xk, pk, name="TLD"):
    return stats.rv_discrete(name=name, values=(xk, pk))


def draw_from_distrib(vals, pdf, size=1):
    """Random Draw from given distribution
    """
    vals = np.array(vals)
    distrib = stats.rv_discrete(values=(range(len(vals)), pdf))
    return vals[distrib.rvs(size=size)]

def AdjustTTHistory(time, temp):
    """Calculate adjusted thermal history

    Useful when one needs to calculate thermal history
    in a borehole when some of the sample reaches
    surface temperature
    """

    def zerointersect(pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2

        xmax = max(x1, x2)
        xmin = min(x1, x2)

        x = np.array([[x1, 1.0], [x2, 1.0]])
        y = np.array([y1, y2])
        A, B = np.linalg.solve(x, y)
        X, Y = -B/A, 0.0
        if(X > xmin and X < xmax):
            return X, Y
        else:
            return None, None

    if(len(time) != len(temp)):
            return "Error"

    TT = [[time[i], temp[i]] for i in range(0, len(time))]

    newTT = []
    for i in range(0, len(TT)-1):
        pt1 = TT[i]
        pt2 = TT[i+1]
        X, Y = zerointersect(pt1, pt2)
        if(X is not None):
            newTT.append([X, Y])

    TT.extend(newTT)
    newTT = []
    for elem in TT:
        if(elem[1] >= 0.0):
            newTT.append(elem)

    newTT.sort()
    newTT.reverse()
    time = [elem[0] for elem in newTT]
    temp = [elem[1] for elem in newTT]
    return time, temp

