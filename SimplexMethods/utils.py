import numpy as np


def _vector_by_index(vec, set):
    del_ind = np.setdiff1d(np.arange(vec.size), set)
    return np.delete(vec, del_ind)


def _matrix_by_index(mat, rows, cows):
    size = mat.shape
    del_rows_ind = np.setdiff1d(np.arange(size[0]), rows)
    del_cols_ind = np.setdiff1d(np.arange(size[1]), cows)
    return np.delete(np.delete(mat, del_rows_ind, 0), del_cols_ind, 1)


def _check_below_zero(vec):
    res = vec < -1e-10
    for r in res:
        if r == False:
            return False
    return True


def _check_above_zero(vec):
    res = vec > 1e-10
    for r in res:
        if r == False:
            return False
    return True


def _check_below_equals_zero(vec):
    res = vec <= 1e-10
    for r in res:
        if r == False:
            return False
    return True


def _check_above_equals_zero(vec):
    res = vec >= -1e-10
    for r in res:
        if r == False:
            return False
    return True

def _check_any_above_zero(vec):
    res = vec > 1e-10
    for r in res:
        if r == True:
            return True
    return False

def _check_any_below_zero(vec):
    res = vec < -1e-10
    for r in res:
        if r == True:
            return True
    return False