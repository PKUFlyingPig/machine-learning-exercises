from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
X = np.random.rand(100).reshape(-1, 1) * np.pi
noise = np.random.normal(size=100) / 20
Y = np.sin(X) + noise

clf = tree.DecisionTreeRegressor()
clf.fit(X, Y)
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph
