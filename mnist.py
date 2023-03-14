# MNIST without hidden layer
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


CSVData = open("H:\\Machine Leanring\\Data\\mnist_train.csv")
x = np.loadtxt(CSVData, delimiter=",")

x = np.divide(x, 255)
y = np.transpose(x)[0]
x = np.transpose(np.transpose(x)[1:])

w = np.zeros((10, 784))
b = np.zeros((10, ))

errorlist = [0.1]

for i in range(200):
    for j in range(x.shape[0]): 
        y0 = np.matmul(w, x[j]) + b
        ytemp = np.zeros((10, ), dtype=int)
        ytemp[int(y[j]*255)] = 1
        er = y0 - ytemp
        dl_dw = 2*np.matmul(np.reshape(er, (10, 1)), np.reshape(x[j], (1, 784)))
        dl_db = 2*er
        w = w - 0.001*dl_dw
        b = b - 0.001*dl_db
    print(i)
    errorlist.append(np.linalg.norm(er)/10)

plt.plot(errorlist)
plt.xlabel('epochs')
plt.ylabel('error')
plt.show()

# for i in range(10):
#     print(np.argmax(np.matmul(w, x[i]) + b))
#     cv.imshow(' ', np.reshape(x[i], (28, 28)))
#     cv.waitKey(0)