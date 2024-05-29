import numpy as np
import matplotlib.pyplot as plt

data = np.array([[7], [10], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23]])
label = np.array([[8], [11], [14],[15], [16], [17], [18], [19], [20], [21], [22], [23], [24]])

plt.figure(figsize=(6, 4))
plt.scatter(data, label, color='blue')
plt.title('Data vs. Label')
plt.xlabel('Data')
plt.ylabel('Label')
plt.grid(True)
plt.show()
