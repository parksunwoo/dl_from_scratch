

import numpy as np
from numpy.core.umath import e


def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
# print(mean_squared_error(np.array(y), np.array(t)))

# y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
# print(mean_squared_error(np.array(y), np.array(t)))


# def cross_entropy_error(y, t):
#
#     delta = 1e-7
#     return -np.sum(t * np.log(y + delta))

# y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
# y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
# print(cross_entropy_error(np.array(y), np.array(t)))

def cross_entropy_error(y, t):

    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshpae(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arrange(batch_size), t])) / batch_size

