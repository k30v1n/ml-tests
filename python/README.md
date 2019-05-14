# NOTES

- [1.DT-hello-world.py](1.DT-hello-world.py) is an example to introduce classifier. Classifiers is just a box of rules where you can run algorithms. In this example the Decision Tree is used to solve some Iris problems.
- [2.DT-iris-visualization.py](2.DT-iris-visualization.py) is an example to print the decision tree into a file so you can check what path the classifier used to make the decision of a given data.
- [3.pipelines.py](3.pipelines.py) is an example showing different classifiers and also calculating its accuracy.

CLASSIFIERS
- The classifiers can be understood as "empty boxes of rules". This objects can run algorithms and make predictions.
- Can also be read as a function "f(x) = y)", X are features (input) and y the labels (output). 
- Can be represented as a line separting different clusters in a graph (this line can be learned)

FEATURES
- Good features are essencials to have a good ML program. But what is a good feature?
    - avoid duplication features
    - if a feature tells you nothing, don't put it. Ex: height in inchs and centimeters, or a caracteristic that is not directly related with what your are analysing
    - the features should be usefull to improve the prediction engine
    - to determine the features try to THINK about what makes sense and what doesn't

MODEL
- it defines the body of a function (predict)
- It can be trained to use different parameters. There is a line that classifier use to distinguish different objects. Looks like this line is somehow controlled by the model. If the model parameters change, the classifier will change too. The model learned, so the predictions.

TESTING
- There is a good pattern that recommends to split the training dataset into two different datasets, Train and Test. The train DS will be used to train the classifier, and the test DS will be used to test and check if the answer is really right. The classifier doesn't know anything about test data.


LINKS
- https://www.youtube.com/playlist?list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal
- http://playground.tensorflow.org 