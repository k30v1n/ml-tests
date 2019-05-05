from sklearn import datasets
iris = datasets.load_iris()

X = iris.data # features
y = iris.target # labels

# this is usually used because a classifier can also be called a mathematical function
# represented on this way 'f(x) = y'.

# split the data into Train data and Test data, half
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

# classifier
# from sklearn import tree
# my_classifier = tree.DecisionTreeClassifier()
from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

# train
my_classifier.fit(X_train, y_train)

# predictions for each X
predictions = my_classifier.predict(X_test)

# check accuracy of each predictions
from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)