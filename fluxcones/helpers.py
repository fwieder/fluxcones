import numpy as np

TOLERANCE = 1e-10


# Support function returns a np.array containing the indices of all entries of a vector larger than the tol
def supp(vector, tol=TOLERANCE):
    return list(np.where(abs(vector) > tol)[0])


# Zero function returns a np.array containing the indices of all entries of a vector smaller than the tol
def zero(vector, tol=TOLERANCE):
    return (np.where(abs(vector) < tol)[0])[0]


# Return the maximal absolute value
def abs_max(vector):
    if all(vector == np.zeros(len(vector))):
        return 0
    abs_max = np.max(np.absolute(vector[vector != 0]))
    return abs_max
