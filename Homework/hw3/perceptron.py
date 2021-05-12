#!/usr/bin/python
#
# CIS 472/572 - Perceptron Template Code
#
# Author: Daniel Lowd <lowd@cs.uoregon.edu>
# Date:   2/9/2018
#
# Please use this code as the template for your solution.
#
import sys
import re
from math import log
from math import exp
import random

MAX_ITERS = 100


# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
        example = [int(x) for x in p.split(l.strip())]
        x = example[0:-1]
        y = example[-1]
        # Each example is a tuple containing both x (vector) and y (int)
        data.append((x, y))
    return (data, varnames)


def sign(x):
    if x > 0:
        return 1
    if x == 0:
        return 0
    else:
        return -1


def calculate(x, w, b):
    return b + sum([x[i] * w[i] for i in range(len(x))])


def update(x, y, w, b):
    return [w[i] + y * x[i] for i in range(len(x))], b + y


# Learn weights using the perceptron algorithm
def train_perceptron(data):

    numvars = len(data[0][0])
    # weights
    w = [0.0] * numvars
    # bias
    b = 0.0
    for i in range(MAX_ITERS):
        for x, y in data:
            res = calculate(x, w, b)
            res = sign(res)
            if res * y <= 0:
                w, b = update(x, y, w, b)

    return w, b


# Compute the activation for input x.
# (NOTE: This should be a real-valued number, not simply +1/-1.)
def predict_perceptron(model, x):
    (w, b) = model

    #
    # YOUR CODE HERE!
    #
    res = b + sum([x[i] * w[i] for i in range(len(x))])
    return sign(res)


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    # Process command line arguments.
    # (You shouldn't need to change this.)
    if (len(argv) != 3):
        print('Usage: perceptron.py <train> <test> <model>')
        sys.exit(2)
    (train, varnames) = read_data(argv[0])
    (test, testvarnames) = read_data(argv[1])
    modelfile = argv[2]

    # Train model
    (w, b) = train_perceptron(train)

    # Write model file
    # (You shouldn't need to change this.)
    f = open(modelfile, "w+")
    f.write('%f\n' % b)
    for i in range(len(w)):
        f.write('%s %f\n' % (varnames[i], w[i]))

    # Make predictions, compute accuracy
    correct = 0
    for (x, y) in test:
        activation = predict_perceptron((w, b), x)
        #print(activation)
        if activation * y > 0:
            correct += 1
    acc = float(correct) / len(test)
    print("Accuracy: ", acc)


if __name__ == "__main__":
    main(sys.argv[1:])
