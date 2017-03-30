import numpy as np
from scipy import spatial


class KMeans:
	def __init__(self, k, max_steps=10**10, assign_points_function="assign_points_eucl", assign_means_function="assign_means_eucl"):
		self.k=k # nr of clusters
		self.max_steps = max_steps
		self.assign_points_function = assign_points_function
		self.assign_means_function = assign_means_function
	
	def fit(self, X):
		"""
		k is the number of clusters
		X is an array where each row is an example data point X[i,:] = example
		"""
		assert type(X) is np.ndarray,				"attribute values X is not ndarray"
		assert len(X.shape)==2, 					"attribute values X doesn't have two dimensions"
		assert X.shape[0]>0,						"attribute values X has no examples"
		assert X.shape[1]>0,						"attribute values X has no attributes"
		if X.dtype!=np.dtype('float64'): 			X = X.astype('float64')
		
		self.nr_exp = X.shape[0]
		self.dim    = X.shape[1]
		self.interval_mins = X.min(axis=0)
		self.interval_maxs = X.max(axis=0)
		self.cluster_centers = self.interval_mins + (self.interval_maxs-self.interval_mins)*np.random.random_sample(self.k * self.dim).reshape(self.k,self.dim)
		
		assign_points		= getattr(self, self.assign_points_function)
		assign_new_means	= getattr(self, self.assign_means_function)

		max_steps, cluster_centers = self.max_steps, self.cluster_centers
		step, change = 0, True
		old_means = cluster_centers.copy() # deep copy
		while change and step<max_steps:
			assignment = assign_points(X, cluster_centers)
			cluster_centers = assign_new_means(assignment, X)
			step+=1
			change = not np.all(old_means==cluster_centers)
			old_means = cluster_centers.copy()
	

	def assign_points_eucl(self, X, cluster_centers):
		distances	= spatial.distance.cdist(X, cluster_centers, metric='euclidean')
		assignment	= np.argmin(distances, axis=1)
		return assignment
	
	def assign_means_eucl(self, assignment, X):
		new_means = np.zeros((self.k, self.dim))
		for i in range(self.k):
			indices			= (assignment==i)
			new_means[i]	= np.mean(X[indices], axis=0)
		return new_means

		

	def print_me(self):
		print "nr_exp"
		print self.nr_exp
		print "dim"
		print self.dim
		print "interval_mins"
		print self.interval_mins
		print "interval_maxs"
		print self.interval_maxs
		print "cluster_centers"
		print self.cluster_centers



X = np.arange(7*5).reshape(7,5)
kmeans = KMeans(2)
kmeans.fit(X)

