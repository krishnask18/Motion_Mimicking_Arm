import numpy as np
import cv2

file = open("H:\\Machine Leanring\\Data\\mnist_train.csv", "r")
img =file.readlines()
file.close()

values = img[-1].split(",")
print(values[0])
image = np.asfarray(values[1:]).reshape(28, 28)
large = cv2.resize(image, fx = 10, fy = 10)
cv2.imshow(" ", large)
cv2.waitKey(0)