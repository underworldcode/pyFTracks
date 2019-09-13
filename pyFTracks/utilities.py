import numpy as np
import pandas as pd
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

def read_mtx_file(filename):
    """ MTX (Madtrax) file parser """
    
    # Check that the extension is actually mtx
    if filename.split(".")[-1] != "mtx":
        raise ValueError("Specify and mtx file")
    
    lines = open(filename, "r").read().splitlines()
    lines = (line.strip() for line in lines)
    
    data = {}
    
    # First Line is the name
    (data["name"]) = next(lines).split(".")[0]
    
    # Skip Second line (not sure what that is)
    next(lines)
    
    # Third line contains count numbers and zeta information
    # nconstraints is the number of boxes defined to constraint
    # thermal history
    
    line2 = next(lines).split()
    nconstraints, ntl, ncounts = (int(val) for val in line2[:3])
    (data["zeta"],
     data["rhod"],
     data["nd"]) = (float(val) for val in line2[3:])

    # Skip the constraints
    for i in range(nconstraints):
        next(lines)
    
    # After the constraints we find:
    # - The Age and associated error
    # - The Mean track length and associated error
    # - The standard deviation and associated error
        
    (data["FTage"],
     data["FTageE"]) = (float(val) for val in next(lines).split())
    (data["MTL"],
     data["MTLE"])   = (float(val) for val in next(lines).split())
    (data["STDEV"],
     data["STDEVE"]) = (float(val) for val in next(lines).split())
    
    # After we find the counts Ns and Ni
    data["Ns"] = []
    data["Ni"] = []
    
    for row in range(ncounts):
        Ns, Ni = (int(val) for val in next(lines).split())
        data["Ns"].append(Ns)
        data["Ni"].append(Ni)
    
    # Finally the track lengths
    data["TL"] = []
    for row in range(ntl):
        data["TL"].append(float(next(lines)))        
        
    return data

def h5store(filename, df, **kwargs):
    store = pd.HDFStore(filename)
    store.put('mydata', df)
    store.get_storer('mydata').attrs.metadata = kwargs
    store.close()

def h5load(store):
    data = store['mydata']
    metadata = store.get_storer('mydata').attrs.metadata
    return data, metadata 
