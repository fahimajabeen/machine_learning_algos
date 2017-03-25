import cython_support as cs

import numpy as np

class DecisionNode:

	def __init__(self, parent_tree, X, Y, depth):
		self.d = depth
		self.node_class = None
		#print "\nnew node\n"

		ld = parent_tree.ld
		if ld.is_leaf(depth, X, Y):
			self.node_class = ld.get_leaf_class(Y)
			#print "class: ", self.node_class
			return

		attrs = parent_tree.attrs
		self.attribute, self.separation_value, self.information_gain = attrs(X, Y)
		
		left_idx  = X[:,self.attribute] <= self.separation_value
		right_idx = X[:,self.attribute] >  self.separation_value
		#print "left right length: ", sum(left_idx), sum(right_idx)
		
		self.lc = DecisionNode(parent_tree, X[left_idx],  Y[left_idx],  depth+1)
		self.rc = DecisionNode(parent_tree, X[right_idx], Y[right_idx], depth+1)


	def classify(self,example):
		if self.node_class != None: return self.node_class
		if example[self.attribute] <= self.separation_value:
			return self.lc.classify(example)
		return self.rc.classify(example)

	def print_me(self, name=""):
		if self.node_class==None:
			if self.lc==None or self.rc==None: return # random subset creates occasional dead nodes
			self.lc.print_me(name+"l")
			self.rc.print_me(name+"r")
		else:
			print name, self.node_class



class FastDecisionNode:
	def __init__(self, parent_tree, X, Y, depth):
		self.d = depth
		self.node_class = None

		ld = parent_tree.ld
		if ld.is_leaf(depth, X, Y):
			self.node_class = ld.get_leaf_class(Y)
			return

		attrs = parent_tree.attrs
		self.attribute, self.separation_value, self.information_gain = attrs(X, Y)
		
		X_left, X_right, Y_left, Y_right = csleft_right_child_data(X, Y, attribute, separation_value)

		self.lc = DecisionNode(parent_tree, X[left_idx],  Y[left_idx],  depth+1)
		self.rc = DecisionNode(parent_tree, X[right_idx], Y[right_idx], depth+1)


	def classify(self,example):
		if self.node_class != None: return self.node_class
		if example[self.attribute] <= self.separation_value:
			return self.lc.classify(example)
		return self.rc.classify(example)

	def print_me(self, name=""):
		if self.node_class==None:
			if self.lc==None or self.rc==None: return # random subset creates occasional dead nodes
			self.lc.print_me(name+"l")
			self.rc.print_me(name+"r")
		else:
			print name, self.node_class







