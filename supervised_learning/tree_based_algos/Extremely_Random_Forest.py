import numpy as np
from tree_components.Decision_Node import DecisionNode
from tree_components.Attribute_Selector import *
from tree_components.Leaf_Checker import LeafDecider
from Extremely_Random_Decision_Tree import ExtremelyRandomDecisionTree
"""
Based on C4.5 - the classifier (it is not for regression)
X contains the attributes that the algorithm can learn from per example
Y are the target values per example
"""
class ExtremelyRandomForest:

	def __init__(self, nr_trees=1, maximum_depth=10**10, min_examples_per_leaf=1):
		self.nr_trees = nr_trees
		self.trees = []
		self.max_depth = maximum_depth
		self.min_examples_per_leaf = min_examples_per_leaf
		

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
		
		for i in range(self.nr_trees):
			curr = ExtremelyRandomDecisionTree(self.max_depth, self.min_examples_per_leaf)
			curr.fit(X,Y)
			self.trees.append(curr)


	def classify(self, example):
		classes = []
		for i in range(self.nr_trees):
			classes.append(self.trees[i].classify(example))
		print "classes: ", classes
		return np.bincount(classes).argmax()

	def get_rules(self, length):
		pass

	def print_me(self):
		for i in range(self.nr_trees):
			print "Tree ", i, ": "
			self.root.print_me()
			print ""




def test_ERF_1():
	rows = 20
	cols = 5
	classes = 4

	X = np.arange(rows*cols).reshape(rows,cols)
	Y = np.arange(rows)
	
	X[:,-1] = X[:,-1] % classes
	Y = Y % classes
	clf = ExtremelyRandomForest(nr_trees=4)
	clf.fit(X,Y)

	example = np.arange(cols)+12*cols
	example[-1] = example[-1] % classes
	assert clf.classify(example)==0, "Error is test_DT_1"
	example = np.arange(cols)+13*cols
	example[-1] = example[-1] % classes
	assert clf.classify(example)==1, "Error is test_DT_1"


def test_ERF_2():
	rows = 20
	cols = 5
	classes = 2
	X = np.arange(rows*cols).reshape(rows,cols)
	Y = np.arange(rows)
	clf = ExtremelyRandomForest(nr_trees=4, maximum_depth=40, min_examples_per_leaf=1)
	clf.fit(X,Y)
	example = np.arange(cols)+12*cols
	assert clf.classify(example)==12, "Error in test_DT_2"
	example = np.arange(cols)+(rows+2)*cols
	assert clf.classify(example)==19, "Error in test_DT_2"



def test_ERF_3():
	from sklearn.datasets import load_iris
	iris = load_iris()
	X, Y = iris["data"], iris["target"]
	clf = ExtremelyRandomForest(nr_trees=7, maximum_depth=5, min_examples_per_leaf=1)
	clf.fit(X,Y)
	counter = 0
	for i in range(len(X)):
		ex = X[i]
		counter += clf.classify(ex)==Y[i]
	if counter<=145: print "count: ", counter
	assert counter>145, "due to randomness this is possible but it is improbable (p<0.01)"
	#clf.print_me()

	
	



if __name__=="__main__":

	for i in range(1000):
		print "counter: ", i
		test_ERF_1()
		test_ERF_2()
		test_ERF_3()




