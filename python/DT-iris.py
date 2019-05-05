import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

# IMPORT A DATASET
iris = load_iris()
testing_items_idx = [0, 50, 100]

# CHECK DATASET
# print "features"
# print iris.feature_names
# print "target/labels"
# print iris.target_names
# print "data"
# print iris.data[0]
# print "target/label 0"
# print iris.target[0]

# TRAIN CLASSIFIER
# first lets remove some data from the dataset, the removed files will be used to
# test our model and check if it is well predict or it is a false positive.

# training items without testing data
train_target = np.delete(iris.target, testing_items_idx)
train_data = np.delete(iris.data, testing_items_idx, axis=0)

# testing data
testing_targets = iris.target[testing_items_idx]
testing_data = iris.data[testing_items_idx]

# train a classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

# TESTING PREDICTION WITH TEST DATA
print testing_targets
print clf.predict(testing_data)

# PREDICT LABEL FOR NEW FLOWER
# VISUALIZE DECISION TREE
from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf,
                        out_file=dot_data,
                        feature_names=iris.feature_names,
                        class_names=iris.target_names,
                        filled=True, rounded=True,
                        impurity=False)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf("temp-DT-iris.pdf")
