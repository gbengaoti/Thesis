import numpy as np
import matplotlib.pyplot as plt

x = np.array([8, 16, 32, 64, 128, 256])

h_acc = np.array([60,61,60,59,59,56])
e_acc = np.array([6,10,13,13,14,17])
hx_acc = np.array([52,56,56,54,56,57])
s_acc = np.array([30,33,34,25,23,26])
c_acc = np.array([21,24,25,21,12,8])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Hidden layer size vs Accuracy(with lr = 0.01)')

ax.set_xlabel('Hidden layer size')
ax.set_ylabel('Validation Accuracy')
ax.plot(x, h_acc, c='r', label='Hydrophobic',linewidth=1.0)
ax.plot(x, e_acc, c='g', label='Electrostatic', linewidth=1.0)
ax.plot(x, hx_acc, c='b', label='Helix',linewidth=1.0)
ax.plot(x, s_acc, c='c', label='Sheet',linewidth=1.0)
ax.plot(x, c_acc, c='m', label='Coil',linewidth=1.0)
plt.legend(bbox_to_anchor=(0.65,0.8),loc=2, borderaxespad=1.)
plt.show()


