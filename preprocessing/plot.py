import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import random

arr = np.array([], dtype=np.float64)
x = np.random.normal(size=1000)

with open(r'data\csv_output\params.csv', 'r') as f:
    reader = csv.reader(f)

    for row in reader:
        arr = np.append(arr, float(row[0]))

print(type(arr[0]))
n, bins, patches = plt.hist(arr, 20, range=(0, 4), density=False, facecolor='g')

plt.xlabel('Wave height')
plt.ylabel('Number of waves')
plt.title('Histogram of wave height')
plt.xlim(0, 4)
plt.show()

