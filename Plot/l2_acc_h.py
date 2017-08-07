import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 0.0001,0.001,0.01,0.1])

h_acc = np.array([0,97,93,86,91])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('L2 regularization vs Accuracy(with lr = 0.001)')

ax.set_xlabel('L2 regularization')
ax.set_ylabel('Validation Accuracy')
ax.plot(x, h_acc, c='r', label='Hydrophobic', marker = '.',linewidth=1.0)

plt.legend(bbox_to_anchor=(0.65,0.9),loc=2, borderaxespad=1.)
plt.show()


