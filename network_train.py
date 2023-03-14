#  784 -->  256 -->   128  --> 64 -->  10 
import numpy as np
import time
import matplotlib.pyplot as plt

def train():
    global w1, w2, w3, w4, b1, b2, b3, b4
    
    w1 = np.random.randn(256, 784)*np.sqrt(1./256)
    w2 = np.random.randn(128, 256)*np.sqrt(1./128)
    w3 = np.random.randn(64, 128)*np.sqrt(1./64)
    w4 = np.random.randn(10, 64)*np.sqrt(1./10)

    b1 = np.random.randn(256, 1)*np.sqrt(1./256)
    b2 = np.random.randn(128, 1)*np.sqrt(1./128)
    b3 = np.random.randn(64, 1)*np.sqrt(1./64)
    b4 = np.random.randn(10, 1)*np.sqrt(1./10)

    train_data = open("H:\\Machine Leanring\\Data\\mnist_train.csv", "r")
    train_images =train_data.readlines()
    train_data.close()

    test_data = open("H:\\Machine Leanring\\Data\\mnist_test.csv", "r")
    test_images =test_data.readlines()
    test_data.close()

    output_nodes = 10
    accur = []
    ep = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    iteration = 15
    for i in range(iteration):
        start = time.time()
        for img in train_images:
            values = img.split(",")
            input = ((np.asfarray(values[1:]) / 255.0 * 0.99) + 0.01)
            targets = np.zeros(output_nodes) + 0.01
            targets[int(values[0])] = 0.99
            output = fwdFeed(w1, w2, w3, w4, b1, b2, b3, b4, input)
            del_w1, del_w2, del_w3, del_w4, del_b1, del_b2, del_b3, del_b4 = backProp(output, targets, input)
            updateWeight(del_w1, del_w2, del_w3, del_w4, del_b1, del_b2, del_b3, del_b4)
        acc = accuracy(train_images)
        accur.append(acc)
        end = time.time()
        print("Epoch No : ")
        print(i+1)
        
        print("Time taken : ")
        print(end - start)
        
        print("Training accuracy : ")
        print(acc)
        print("\n\n")
    test_accur = accuracy(test_images)
    print("Test accuracy: ", test_accur)
    plt.scatter(1, test_accur, c="red")
    plt.scatter(ep, accur)
    plt.plot(ep, accur)
    plt.show()

def backProp(output, targets, x):
    global w1, w2, w3, w4, b1, b2, b3, b4
    global a1, h1, a2, h2, a3, h3, a4, y

    error = 2 * (output - targets).diagonal() / output.shape[0] * softmax(a4, derivative=True)
    del_w4 = np.outer(error, h3)
    del_b4 = error

    error = np.dot(w4.transpose(), error) * sigmoid(a3, derivative=True)
    del_w3 = np.outer(error, h2)
    del_b3 = error

    error = np.dot(w3.transpose(), error) * sigmoid(a2, derivative=True)
    del_w2 = np.outer(error, h1)
    del_b2 = error

    error = np.dot(w2.transpose(), error) * sigmoid(a1, derivative=True)
    del_w1 = np.outer(error, x)
    del_b1 = error

    return del_w1, del_w2, del_w3, del_w4, del_b1, del_b2, del_b3, del_b4

def fwdFeed(w1, w2, w3, w4, b1, b2, b3, b4, x):
    global a1, h1, a2, h2, a3, h3, a4, y
    a1 = (np.dot(w1, x) + b1).diagonal()
    h1 = sigmoid(a1)

    a2 = (np.dot(w2, h1) + b2).diagonal()
    h2 = sigmoid(a2)

    a3 = (np.dot(w3, h2) + b3).diagonal()
    h3 = sigmoid(a3)

    a4 = (np.dot(w4, h3) + b4).diagonal()
    y = softmax(a4)

    return y

def updateWeight(del_w1, del_w2, del_w3, del_w4, del_b1, del_b2, del_b3, del_b4):
    global w1, w2, w3, w4, b1, b2, b3, b4
    lr = 0.25

    w1 = w1 - lr*del_w1
    w2 = w2 - lr*del_w2
    w3 = w3 - lr*del_w3
    w4 = w4 - lr*del_w4

    b1 = b1 - lr*del_b1
    b2 = b2 - lr*del_b2
    b3 = b3 - lr*del_b3
    b4 = b4 - lr*del_b4

def sigmoid(x, derivative = False):
    if derivative:
        return (np.exp(-x))/((np.exp(-x)+1)**2)
    return 1/(1 + np.exp(-x))

def softmax(x, derivative = False):
    exps = np.exp(x - x.max())
    if derivative:
        return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
    return exps / np.sum(exps, axis=0)

def accuracy(train_list):
     predictions = []

     for x in train_list:
          all_values = x.split(',')
          inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
          targets = np.zeros(10) + 0.01
          targets[int(all_values[0])] = 0.99
          output = fwdFeed(w1, w2, w3, w4, b1, b2, b3, b4, inputs)
          pred = np.argmax(output)
          predictions.append(pred == np.argmax(targets))
      
     return np.mean(predictions)

train()