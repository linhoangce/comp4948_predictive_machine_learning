import numpy as np
from matplotlib import pyplot as plt

weights = [0.5, 2.3, 2.9]
heights = [1.4, 1.9, 3.2]

def getSlopeOfLossFunction(weights, heights, intercept):
    sum = 0
    BETA = 0.64
    for i in range(0, len(weights)):
        sum += -2 * (heights[i] - intercept - BETA * weights[i])

    print("Intercept: " + str(intercept) + " Res: " + str(round(sum, 2)))
    return sum

intercept = 0.95
getSlopeOfLossFunction(weights, heights, intercept)

intercepts = np.arange(start=0, stop=1, step=0.1)
results = [getSlopeOfLossFunction(weights, heights, i)
           for i in intercepts]

plt.plot(intercepts, results)
plt.show()