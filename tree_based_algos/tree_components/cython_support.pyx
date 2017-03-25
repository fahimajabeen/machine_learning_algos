"""
cython_support.pyx

this file covers two things:
	1) Some cython functions meant to speed up python a bit
	2) It provides an interface to a C-file which allows very fast programming

"""
import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np


# declare the interface to the C code
cdef extern int c_smaller_greater(double* X, int* idx, int n, double splitVal)
cdef extern void c_copy_column(double* X, double* X_column, int nr_rows, int nr_cols, int col_idx)
cdef extern void c_create_X_children(double* X, int* Y, double* X_left, double* X_right, int* Y_left, int* Y_right, int* idx_array, int split_idx, int nr_rows, int nr_cols)

def smaller_greater(np.ndarray[double, ndim=1, mode="c"] X not None, np.ndarray[int, ndim=1, mode="c"] idx not None, double splitVal):
	cdef int n, split_idx
	n = int(len(X))
	split_idx = c_smaller_greater(&X[0], &idx[0], n, splitVal)
	return split_idx


def copy_column(np.ndarray[double, ndim=2, mode="c"] X not None, np.ndarray[double, ndim=1, mode="c"] X_column, long col_idx):
	cdef int nr_rows, nr_cols
	nr_rows, nr_cols = X.shape[0], X.shape[1]
	c_copy_column(&X[0,0], &X_column[0], nr_rows, nr_cols, int(col_idx))
	return

def copy_column2(np.ndarray[double, ndim=2, mode="c"] X not None, long col_idx):
	cdef int nr_rows, nr_cols
	nr_rows, nr_cols = X.shape[0], X.shape[1]
	cdef np.ndarray[double, ndim=1, mode="c"] X_column = np.empty(nr_rows, dtype=np.double)
	c_copy_column(&X[0,0], &X_column[0], nr_rows, nr_cols, int(col_idx))
	return X_column


def left_right_child_data(np.ndarray[double, ndim=2, mode="c"] X not None, np.ndarray[int, ndim=1, mode="c"] Y not None, int attr_idx, double splitVal):
	cdef int nr_rows, nr_cols, split_idx
	nr_rows, nr_cols = X.shape[0], X.shape[1]
	
	# cython version of X_column = X[:,attr_idx]
	cdef np.ndarray[double, ndim=1, mode="c"] X_column = np.empty(nr_rows, dtype=np.double)
	c_copy_column(&X[0,0], &X_column[0], nr_rows, nr_cols, attr_idx)

	# index_array so that: index_array[:split_idx] indices with X_column[index_array[:split_idx] <= splitVal
	cdef np.ndarray[int, ndim=1, mode="c"] idx_array = np.empty(nr_rows, dtype=np.int32)
	split_idx = c_smaller_greater(&X_column[0], &idx_array[0], nr_rows, splitVal)
	
	
	# create X's children
	cdef np.ndarray[double, ndim=2, mode="c"] X_left   = np.empty(shape=(split_idx,nr_cols), dtype=np.double)
	cdef np.ndarray[double, ndim=2, mode="c"] X_right  = np.empty(shape=((nr_rows-split_idx),nr_cols), dtype=np.double)
	cdef np.ndarray[int, ndim=1, mode="c"]   Y_left   = np.empty(split_idx, dtype=np.int32)
	cdef np.ndarray[int, ndim=1, mode="c"]   Y_right  = np.empty((nr_rows-split_idx), dtype=np.int32)

	# TODO - this throws Out of bounds on buffer access (axis 0) when arrays are of length ZERO
	c_create_X_children(&X[0,0], &Y[0], &X_left[0,0], &X_right[0,0], &Y_left[0], &Y_right[0], &idx_array[0], split_idx, nr_rows, nr_cols)
	
	return X_left, X_right, Y_left, Y_right
	





















