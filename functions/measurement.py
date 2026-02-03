from LightPipes import Field
from LightPipes import Normal

import numpy as np
import math

def overlap_integral(F: Field,G: Field) -> float:
    """
    Calculates a numerical value for overlap of two LightPipes fields. To be used as a point of comparison, is not the fully correct overlap integral value?
    
    :param F: Input Field 1
    :type F: Field
    :param G: Input Field 2
    :type G: Field
    :return: Overlap Integral
    :rtype: float
    """
    F,G=Normal(F),Normal(G)
    Ffield,Gfield=np.conjugate(F.field),G.field
    fieldArr=np.multiply(Ffield,Gfield)
    summed=abs(np.sum(fieldArr))**2
    return summed

# Don't know what this does
def normTomography(ints, d):
    return np.concatenate([
        chunk / chunk.sum() if chunk.sum() != 0 else chunk
        for chunk in np.split(ints, range(d, len(ints), d))
    ])

#Crosstalk of two vectors
def crosstalkVecs(Fs,Gs):
    c,C=[],[]
    for F in Fs:
        c=[]
        for G in Gs:
            c.append(abs(np.dot(np.conjugate(F),G))**2)
        C.append(c/sum(c))
    return C

#Calculate full crosstalk of two beam lists
def crosstalk(Fs,Gs):
    c,C=[],[]
    for F in Fs:
        c=[]
        for G in Gs:
            c.append(overlap_integral(F,G))
        C.append(c/sum(c))
    return C

def tomography(Fs,Gs):
    c,C=[],[]
    for F in Fs:
        c=[]
        for G in Gs:
            c.append(overlap_integral(F,G))
        C.append(normTomography(c,math.isqrt(len(Fs))))
    return C

def beamsError(Fs,Gs):
    array=crosstalk(Fs,Gs)
    error=1-np.mean([diag[i] for i,diag in enumerate(array)])
    return error