import numpy as np
import random
#from keras.datasets import mnist as dataset
from keras.datasets import cifar10 as dataset
#from keras.datasets import imdb as dataset

# perception model, a single layer NN
# model --> perception model table, 
#    includes: 
#       "weights"(coefficient, size nf*nl)
#       "biases" (size nl)
#       "alpha" (learning rate)

def perceptron_predict(model,data,bias_on = False):
    # calculate the values and select the largest one as predicted label
    labels = np.matmul(model["weights"],data)
    if bias_on:
        labels += model["biases"]
    return np.argmax(labels)

def update_perceptron(model,data,label,bias_on = False):
    # update perceptron model if prediction is not correct
    # the weight(bias) of predicted: minus data(label) * learning rate
    # the weight(bias) of true label: plus data(label) * learning rate
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
        index = random.sample(index, k=n_data)
        random.shuffle(index)
        for jj in index:
            model,updated = update_perceptron(model,data[jj].flatten(),label[jj],bias_on)
            if updated:
                count+=1
        if (ii+1)%print_step == 0:
            print("epoch = %d, update count %d" %(ii+1,count))
    return model

# load data
(train_X, train_y), (test_X, test_y) = dataset.load_data()
# total training steps 
epochSteps = 50
# learning rate
learnRate = 1
# log steps
printSteps = 10
# iteration
iteration = 10

# model without biases
acc_no_bias = 0
for ii in range(iteration):
    label,image = train_y, train_X

    n_label = len(np.unique(label))
    n_feature = len(image[0].flatten())

    model = {}
    model["weights"] = np.random.rand(n_label,n_feature)
    model["biases"] = np.random.rand(n_label)
    model["alpha"] = learnRate

    model = train_model(model,image,label, epoch=epochSteps, print_step = printSteps, bias_on = False)
    label,image = test_y, test_X
    acc_no_bias += test_model(model,image,label, bias_on = False)

# model with biases
acc_bias = 0
for ii in range(iteration):
    label,image = train_y, train_X

    n_label = len(np.unique(label))
    n_feature = len(image[0].flatten())

    model = {}
    model["weights"] = np.random.rand(n_label,n_feature)
    model["biases"] = np.random.rand(n_label)
    model["alpha"] = learnRate

    model = train_model(model,image,label, epoch = epochSteps, print_step = printSteps, bias_on = True)
    label,image = test_y, test_X
    acc_bias += test_model(model,image,label, bias_on = True)

print(acc_bias/10, acc_no_bias/10)
