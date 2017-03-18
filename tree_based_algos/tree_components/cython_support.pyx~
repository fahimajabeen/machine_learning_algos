import numpy as np
cimport numpy as np
import cython
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef long[:] deepcopy(long[:] in_a):
	cdef long len_out = len(in_a)
	cdef long i
	cdef long[:] out_a = np.empty(len_out, dtype=long)
	for i in range(len_out):
		out_a[i] = in_a[i]
	return np.asarray(out_a)
	

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef quicksort(long [:] a):
	cdef long n = long(len(a)-1)
	quicksort_call(a, 0, n)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void quicksort_call(long [:] a, long lo, long hi):
	cdef long p
	if lo<hi:
		p = partition(a, lo, hi)
		quicksort_call(a, lo, p-1)
		quicksort_call(a, p+1, hi)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef long partition(long [:] a, long lo, long hi):
	cdef long pivot = a[hi]
	cdef long i = lo - 1
	cdef int j
	for j in range(lo, hi):
		if a[j] <= pivot:
			i += 1
			a[i], a[j] = a[j], a[i]
	a[i+1], a[hi] = a[hi], a[i+1]
	return i+1


#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef unique(long [:] a):
	#returns all unique elements in a
	#returns the indices for a mapping to the unique element array 	
	cdef long[:] b = deepcopy(a)
	quicksort(b)
	unique = [b[0]]
	cdef int i
	cdef long last = b[0]
	cdef int n = len(a)
	for i in range(n):
		if last != b[i]:
			last = b[i]
			unique.append(last)
	unique = np.array(unique)
	return unique


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef split_by_attribute(double[:] X_attr, long[:] Y, double split_val):
	cdef int len_smaller = 0
	cdef int i, j, k = 0
	cdef int Y_len = len(Y)
	for i in range(Y_len):
		if X_attr[i] <= split_val:
			len_smaller += 1
	cdef int len_larger  = Y_len - len_smaller
	
	cdef np.ndarray[long, ndim=1, mode="c"] Y_smaller = np.empty(len_smaller, dtype=np.int64)
	cdef np.ndarray[long, ndim=1, mode="c"] Y_larger  = np.empty(len_larger,  dtype=np.int64)

	cdef long* Y_sma = &Y_smaller[0]
	cdef long* Y_lar = &Y_larger[0]
	
	for i in range(Y_len):
		if X_attr[i] <= split_val:
			Y_sma[j] = Y[i]
			j += 1
		else:
			Y_lar[k] = Y[i]
			k += 1

	return Y_smaller, Y_larger, len_smaller, len_larger
















