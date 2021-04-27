#!/usr/bin/python3
#
# CIS 472/572 -- Programming Homework #1
#
# Starter code provided by Daniel Lowd, 1/25/2018
# Author:Xing Qian
#
import sys
import re
# Node class for the decision tree
import node
import math

train = None
varnames = None
test = None
testvarnames = None
root = None
have_used_feat = []

columns = []


# Helper function computes entropy of Bernoulli distribution with
# parameter p
def entropy(p: float):
    # >>>> YOUR CODE GOES HERE <<<<
    # For now, always return "0":
    if p == 0 or p == 1:
        return 0

    p1, p2 = p, 1 - p
    en = -p1 * math.log(p1, 2) - p2 * math.log(p2, 2)
    return en


# Compute information gain for a particular split, given the counts
# py_pxi : number of occurences of y=1 with x_i=1 for all i=1 to n
# pxi : number of occurrences of x_i=1
# py : number of ocurrences of y=1
# total : total length of the data
def infogain(py_pxi, pxi, py, total):
    # >>>> YOUR CODE GOES HERE <<<<
    # For now, always return "0":
    if py != total:
        base_entropy = entropy(py / total)
    else:
        base_entropy = 0

    x_1, x_0 = pxi, total - pxi
    if x_1 != 0:
        base_entropy -= (x_1 / total) * entropy(py_pxi / x_1)
    if x_0 != 0:
        base_entropy -= (x_0 / total) * entropy((py - py_pxi) / x_0)
    return base_entropy


# OTHER SUGGESTED HELPER FUNCTIONS:
# - collect counts for each variable value with each class label
# - find the best variable to split on, according to mutual information
# - partition data based on a given variable
def counts(sub_data: list, label):
    total_length = len(label)
    py = label.count(1)
    pxi = sub_data.count(1)
    py_pxi = 0
    for index in range(total_length):
        if sub_data[index] == label[index] == 1:
            py_pxi += 1
    return infogain(py_pxi, pxi, py, total_length)


def choose_best_feature_split(data, label):
    global have_used_feat, columns
    feature_number = len(data[0]) - 1
    best_info_gain = 0.0
    best_feature = -1
    for index in range(feature_number):
        if index in have_used_feat:
            continue
        feat_list = [sample[index] for sample in data]
        info_gain = counts(feat_list, label)
        if columns[index] in ['gill-spacing-close', 'gill-spacing-crowded']:
            print(columns[index] , '---', info_gain)
        if info_gain >= best_info_gain:
            best_info_gain = info_gain
            best_feature = index
    # if best_info_gain < 0.8:
    #     return -1
    return best_feature, best_info_gain


def split_data(data, index, value):
    sub = []
    for row in data:
        if row[index] == value:
            sub.append(row)
    return sub


def major(labels):
    count1 = labels.count(1)
    count0 = labels.count(0)
    return count1 if count1 > count0 else count0


# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
        data.append([int(x) for x in p.split(l.strip())])
    return (data, varnames)


# Saves the model to a file.  Most of the work here is done in the
# node class.  This should work as-is with no changes needed.
def print_model(root, modelfile):
    f = open(modelfile, 'w+')
    root.write(f, 0)


# Build tree in a top-down manner, selecting splits until we hit a
# pure leaf or all splits look bad.
def build_tree(data, varnames):
    # >>>> YOUR CODE GOES HERE <<<<
    # For now, always return a leaf predicting "1":
    global columns
    columns = varnames
    if len(data) == 0:
        return node.Leaf(varnames, 1)
    label = [sample[-1] for sample in data]
    if len(set(label)) == 1:
        return node.Leaf(varnames, label[0])
    if len(data[0]) == 1:
        return node.Leaf(varnames, major(label))
    best_feat_index, _ = choose_best_feature_split(data, label)
    print(f'select feature is {columns[best_feat_index]}, it is info gain is {_}')
    #have_used_feat.append(best_feat_index)
    tree = node.Split(names=varnames, var=best_feat_index,
                      left=build_tree(split_data(data, index=best_feat_index, value=0), varnames),
                      right=build_tree(split_data(data, index=best_feat_index, value=1), varnames))
    return tree


# "varnames" is a list of names, one for each variable
# "train" and "test" are lists of examples.
# Each example is a list of attribute values, where the last element in
# the list is the class value.
def loadAndTrain(trainS, testS, modelS):
    global train
    global varnames
    global test
    global testvarnames
    global root
    (train, varnames) = read_data(trainS)
    (test, testvarnames) = read_data(testS)
    modelfile = modelS

    # build_tree is the main function you'll have to implement, along with
    # any helper functions needed.  It should return the root node of the
    # decision tree.
    root = build_tree(train, varnames)
    print_model(root, modelfile)


def runTest():
    correct = 0
    # The position of the class label is the last element in the list.
    yi = len(test[0]) - 1
    for x in test:
        # Classification is done recursively by the node class.
        # This should work as-is.
        pred = root.classify(x)
        if pred == x[yi]:
            correct += 1
    acc = float(correct) / len(test)
    return acc


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 3):
        print('Usage: python3 id3.py <train> <test> <model>')
        sys.exit(2)
    loadAndTrain(argv[0], argv[1], argv[2])

    acc = runTest()
    print("Accuracy: ", acc)


if __name__ == "__main__":
    main(sys.argv[1:])
