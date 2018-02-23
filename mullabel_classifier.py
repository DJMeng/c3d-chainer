from chainer import links as L
from chainer import functions as F
import numpy as np


class Mullabel_Classifier(L.Classifier):
	
	@staticmethod
	def sigmoid_cross_entropy_mullabel(x, t):
		return F.sigmoid_cross_entropy(x, t)
		
	@staticmethod
	def accuracy_mullable(x, t):
		x = [list(map(lambda n: 0 if n < 0.5 else 1, l)) for l in x.data]
		size = len(t)
		acc = 0.0
		for i, label in enumerate(t):
			if np.array_equal(label, x[i]):
				acc += 1
		return acc / size
	
	def __init__(self, predictor):
		super(Mullabel_Classifier, self).__init__(predictor)
		self.lossfun = Mullabel_Classifier.sigmoid_cross_entropy_mullabel
		self.accfun = Mullabel_Classifier.accuracy_mullable
