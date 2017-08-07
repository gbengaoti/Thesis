import numpy as np
import matplotlib.pyplot as plt

x = np.arange(500)
y = np.genfromtxt('convergence_first_model.txt')

y = np.round(y,4)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('No of Epochs')
ax.set_ylabel('MSE')
ax.plot(x, y, c='r', label='keras', linewidth=1.0)
plt.legend()
plt.show()


