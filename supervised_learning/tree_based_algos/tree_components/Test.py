import numpy as np
import time

import cython_support as cs

def Test_CS_1():
	a = np.arange(15, dtype=float)
	for i in range(5):
		a[len(a)-i-1]=i
	print "A:\n", a	
	idx_a = np.empty(a.shape, dtype=np.int32)
	splitVal = 4.0
	split_idx = cs.smaller_greater(a, idx_a, splitVal)
	print "Split idx & Idx A:\n", split_idx, idx_a
	print "A smaller:\n", a[idx_a[:split_idx]]
	print "A larger:\n", a[idx_a[split_idx:]]


def Test_CS_2():
	n, m = 10,10
	X = np.arange(n*m, dtype=float).reshape(n,m)
	X_col = np.empty(n, np.double)
	cs.copy_column(X, X_col, 4)
	print X
	print X_col

def Test_CS_3():
	n, m = 10,10
	X = np.arange(n*m, dtype=float).reshape(n,m)
	X_col = cs.copy_column2(X, 4)
	print X
	print X_col

def Test_CS_4():
	print "Test CS 4"
	n, m = 10,10
	X = np.arange(n*m, dtype=float).reshape(n,m)
	X[7,4] = 10
	Y = np.arange(n, dtype=np.int32)
	attr_idx = 4
	split_val = 15
	
	xl,xr,yl,yr = cs.left_right_child_data(X, Y, attr_idx, split_val)
	print xl
	print xr
	print yl
	print yr

def Test_CS_5():
	print "Test CS 5"
	n, m = 1000,4000
	X = np.arange(n*m, dtype=float).reshape(n,m)
	X[7,4] = 10
	X = X%(n-1)
	Y = np.arange(n, dtype=np.int32)
	attr_idx = 4
	split_val = 15.0
	
	st = time.clock()
	xl,xr,yl,yr = cs.left_right_child_data(X, Y, attr_idx, split_val)
	print "fast time: ", time.clock()-st
	
	st = time.clock()
	left_idx  = X[:,attr_idx] <= split_val
	right_idx = X[:,attr_idx] >  split_val
	xl, yl    = X[left_idx],  Y[left_idx]
	xr, yr    = X[right_idx], Y[right_idx]
	print "slow time: ", time.clock()-st


if __name__=="__main__":
	
	Test_CS_1()
	Test_CS_2()
	Test_CS_3()
	Test_CS_4()
	Test_CS_5()





























