import numpy as np
import random
from keras.datasets import mnist as dataset
#from keras.datasets import cifar10 as dataset
#from keras.datasets import imdb as dataset
# Naive Bayes:
#   1. assumes all features are independent
#   2. use Bayes rule: p(a|b) * p(b) = p(b|a) * p(a)

def train_by_count(model,data,label):
    for i in range(len(label)):
        model["py"][label[i]] += 1
        model["pxy"][label[i]] += data[i].flatten()

def NB_predict(model,data):
    labels = model["py"] + np.matmul(model["pxy"],data.flatten())
    return np.argmax(labels)

def test_model(model,data,label):
    n_data = len(label)
    count = 0.0
    for ii in range(n_data): 
        if label[ii] == NB_predict(model,data[ii]):
            count+=1
    accuracy = 1.0*count/n_data*100
    print("accuracy is: %.2f %% " %(accuracy))
    return accuracy

# load data
(train_X, train_y), (test_X, test_y) = dataset.load_data()
iteration = 1

# model without biases

for ii in range(iteration):
    label,image = train_y, train_X

    n_label = len(np.unique(label))
    n_feature = len(image[0].flatten())
    model = {}
    model["pxy"] = np.ones((n_label,n_feature))
    model["py"] = np.ones(n_label)
    train_by_count(model, image, label)
    test_model(model, image, label)
    
