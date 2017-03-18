import numpy as np
from tree_components.Decision_Node import DecisionNode
from tree_components.Attribute_Selector import *
from tree_components.Leaf_Checker import LeafDecider
"""
Based on C4.5 - the classifier (it is not for regression)
X contains the attributes that the algorithm can learn from per example
Y are the target values per example
"""
class DecisionTree:
	def __init__(self, maximum_depth=10**10, min_examples_per_leaf=1):
		self.n_exp  = 0	# nr of examples
		self.n_attr = 0	# nr of attributes
		self.ld		= LeafDecider(maximum_depth=maximum_depth, min_examples_per_leaf=min_examples_per_leaf)
		self.attrs	= AttributeSelector("standard decision tree").get_attribute_selector()

	def fit(self, X=np.zeros((0,0)), Y=np.zeros(0)):
		# assertions should check whether the tree can be built
		# assertions concerning X
		assert type(X) is np.ndarray,	"attribute values X is not ndarray"
		assert len(X.shape)==2, 		"attribute values X has more that two dimensions"
		assert X.shape[0]>0,			"attribute values X has no examples"
		assert X.shape[1]>0,			"attribute values X has no attributes"
		# assertions concerning Y
		assert type(Y) is np.ndarray,	"target values Y is not ndarray"
		assert len(Y.shape)==1,			"target values Y has more that two dimensions"
		assert Y.shape[0]==X.shape[0],	"target values Y has an invalid amount of examples"
		
		self.n_exp  = X.shape[0]	# nr of examples
		self.n_attr = X.shape[1]-1	# nr of attributes
		self.root = DecisionNode(self, X, Y, depth=0)
		

	def classify(self, example):
		return self.root.classify(example)

	def get_rules(self, length):
		pass

	def print_me(self):
		self.root.print_me()
	


def test_DT_1():
	"""
	-creates a tree with 100 classes and one example per class
	-this leads to 100 leaves
	-when classifying an arange in the bottom, arange(cols) + q*cols
	leads to class q
	"""
	rows = 20
	cols = 5
	classes = 4

	X = np.arange(rows*cols).reshape(rows,cols)
	Y = np.arange(rows)
	
	X[:,-1] = X[:,-1] % classes
	Y = Y % classes
	clf = DecisionTree()
	clf.fit(X,Y)

	example = np.arange(cols)+12*cols
	example[-1] = example[-1] % classes
	assert clf.classify(example)==0, "Error is test_DT_1"
	example = np.arange(cols)+13*cols
	example[-1] = example[-1] % classes
	assert clf.classify(example)==1, "Error is test_DT_1"


def test_DT_2():
	"""
	-creates a tree with 100 classes and one example per class
	-this leads to 100 leaves
	-when classifying an arange in the bottom, arange(cols) + q*cols
	leads to class q
	"""
	rows = 20
	cols = 5
	classes = 2
	X = np.arange(rows*cols).reshape(rows,cols)
	Y = np.arange(rows)
	clf = DecisionTree(maximum_depth=40, min_examples_per_leaf=1)
	clf.fit(X,Y)
	example = np.arange(cols)+12*cols
	assert clf.classify(example)==12, "Error in test_DT_2"
	example = np.arange(cols)+(rows+2)*cols
	assert clf.classify(example)==19, "Error in test_DT_2"



def test_DT_3():
	from sklearn.datasets import load_iris
	iris = load_iris()
	X = iris.data[:,:]
	Y = iris.target
	clf = DecisionTree(maximum_depth=5, min_examples_per_leaf=1)
	clf.fit(X,Y)
	for i in range(len(X)):
		assert clf.classify(X[i]) == Y[i], "The decision tree should get all right, something's wrong."
	#clf.print_me()

if __name__=="__main__":

	for i in range(1000):
		print "counter: ", i
		test_DT_1()
		test_DT_2()
		test_DT_3()






