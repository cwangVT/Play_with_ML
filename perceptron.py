import os
import numpy as np
import struct
import random
from read_MNIST import read_mnist
# implement
def perceptron_predict(model,data,bias_on = False):
    labels = np.matmul(model["weights"],data)
    if bias_on:
        labels += model["biases"]
    return np.argmax(labels)

def update_perceptron(model,data,label,bias_on = False):
    predicted = perceptron_predict(model,data)
    if predicted != label:
        model["weights"][predicted] -= model["alpha"]*data
        model["weights"][label] += model["alpha"]*data
        if bias_on:
            model["biases"][predicted] -= model["alpha"]*label
            model["biases"][label] += model["alpha"]*label
        return model,True
    return model,False

def test_model(model,data,label,bias_on = False):
    n_data = len(label)
    count = 0.0
    for ii in range(n_data):
        if label[ii] == perceptron_predict(model,data[ii].flatten(), bias_on):
            count+=1
    accuracy = 1.0*count/n_data*100
    print("accuracy is: %.2f %% " %(accuracy))
    return accuracy


def train_model(model,data,label, epoch, print_step = 50, bias_on = False):
    n_data = len(label)

    for ii in range(epoch):
        count = 0
        index = range(n_data)
        random.shuffle(index)
        for jj in index:
            model,updated = update_perceptron(model,data[jj].flatten(),label[jj],bias_on)
            if updated:
                count+=1
        if (ii+1)%print_step == 0:
            print "epoch = %d, update count %d" %(ii+1,count)
    return model


acc_no_bias = 0
for ii in range(10):
    label,image = read_mnist()

    n_label = len(np.unique(label))
    n_feature = len(image[0].flatten())

    model = {}
    model["weights"] = np.random.rand(n_label,n_feature)
    model["biases"] = np.random.rand(n_label)
    model["alpha"] = 1

    model = train_model(model,image,label, epoch=50, bias_on = False)
    label,image = read_mnist(task = "testing")
    acc_no_bias += test_model(model,image,label, bias_on = False)

acc_bias = 0
for ii in range(10):
    label,image = read_mnist()

    n_label = len(np.unique(label))
    n_feature = len(image[0].flatten())

    model = {}
    model["weights"] = np.random.rand(n_label,n_feature)
    model["biases"] = np.random.rand(n_label)
    model["alpha"] = 1

    model = train_model(model,image,label, epoch=50, bias_on = True)
    label,image = read_mnist(task = "testing")
    acc_bias += test_model(model,image,label, bias_on = True)

print acc_bias/10, acc_no_bias/10
