import numpy as np
import matplotlib.pylab as plt

# step1
# def step_function(x):
#     if x > 0:
#         return 1
#     else:
#         return 0

# step2
# def step_function(x):
#     y = x > 0
#     return y.astype(np.int)

# import numpy as np
# x = np.array([-1.0, 1.0, 2.0])
# x
#
# y = x > 0
# y


# step3
def step_function(x):
    return np.array(x > 0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()



