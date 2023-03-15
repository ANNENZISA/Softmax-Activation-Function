import matplotlib.pyplot as plt
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

x = np.array([1, 2, 3, 4, 5])
y = softmax(x)

plt.plot(x, y, 'o')
plt.title('Softmax Activation Function')
plt.xlabel('Input')
plt.ylabel('Output Probability')
plt.show()
