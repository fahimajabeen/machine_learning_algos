import cython_support as cs
import numpy as np
import time

def test_CS_1():
	a = np.random.randint(30, size=5000)
	b = np.copy(a)
	b[5] = -10
	c = cs.deepcopy(a)
	c[5] = -10
	for i in range(len(a)):
		assert b[i]==c[i], "test CS 1 is wrong"
	assert a[5]!=b[5], "test CS 1 is wrong"
	assert b[5]==c[5], "test CS 1 is wrong"

def test_CS_2():
	a = np.random.randint(30, size=10000)
	b = cs.deepcopy(a)
	cs.quicksort(b)
	c = np.sort(a)
	for i in range(len(b)):
		assert b[i]==c[i], "test CS 2 (quicksort) is wrong"

#TODO make sure this works
def test_CS_3():
	X = np.arange(20*5).reshape(20, 5).astype(float)
	X[:,2] = X[:,2] % 4
	Y = np.arange(20) % 4
	ys, yl, lys, lyl = cs.split_by_attribute(X[:,2], Y, 2)
	print X
	print len(Y), Y
	print len(ys), lys, ys
	print len(yl), lyl, yl


if __name__=="__main__":
	test_CS_1()
	test_CS_2()
	test_CS_3()




