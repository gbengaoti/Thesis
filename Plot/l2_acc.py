import numpy as np
import matplotlib.pyplot as plt

x = np.array([0.0001,0.001,0.01,0.1])

h_acc = np.array([58,59,59,59])
e_acc = np.array([5,5,5,4])
hx_acc = np.array([58,58,59,57])
s_acc = np.array([34,34,32,34])
c_acc = np.array([23,23,23,22])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('L2 regularization vs Accuracy(with lr = 0.01)')

ax.set_xlabel('L2 regularization')
ax.set_ylabel('Validation Accuracy')
ax.plot(x, h_acc, c='r', label='Hydrophobic', marker = '.',linewidth=1.0)
ax.plot(x, e_acc, c='g', label='Electrostatic', marker = '.', linewidth=1.0)
ax.plot(x, hx_acc, c='b', label='Helix',marker = '.', linewidth=1.0)
ax.plot(x, s_acc, c='c', label='Sheet',marker = '.',linewidth=1.0)
ax.plot(x, c_acc, c='m', label='Coil',marker = '.',linewidth=1.0)
plt.legend(bbox_to_anchor=(0.65,0.9),loc=2, borderaxespad=1.)
plt.show()


