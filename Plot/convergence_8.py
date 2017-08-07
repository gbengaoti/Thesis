import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1000)
train = np.genfromtxt('convergence_first_model8units_train.txt')
valid = np.genfromtxt('convergence_first_model8units_valid.txt')
train = np.round(train,4)
valid = np.round(valid,4)
labels = ['train', 'valid']
colors = ['r','g']

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('No of Epochs')
ax.set_ylabel('MSE')
ax.plot(x, train,  c='r', label='train', linewidth=1.0)
ax.plot(x, valid,  c='g', label='valid', linewidth=1.0)
plt.legend()
plt.show()


