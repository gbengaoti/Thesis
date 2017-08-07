import numpy as np
import matplotlib.pyplot as plt

x = np.array([8, 16, 32, 64, 128, 256])

h_acc = np.array([97,61,60,59,59,56])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Hidden layer size vs Accuracy(with lr = 0.001)')

ax.set_xlabel('Hidden layer size')
ax.set_ylabel('Validation Accuracy')
ax.plot(x, h_acc, c='r', label='Hydrophobic',linewidth=1.0)

plt.legend(bbox_to_anchor=(0.65,0.8),loc=2, borderaxespad=1.)
plt.show()


