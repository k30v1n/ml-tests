from sklearn import tree

### COLLECT TRAINING DATA ###

# sklearn uses real numbers instead of strings

# 0 = smooth
# 1 = bumpy
features = [[140,1], [130, 1], [150, 0], [170, 0]]

# 0 = apple
# 1 = orange
labels = [0, 0, 1, 1]


### TRAIN A CLASSIFIER (decision tree) ###

# empty box of rules (without the learning algorithm)
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(features, labels)

### MAKE PREDICTIONS ###
print(classifier.predict([[140,1]]))