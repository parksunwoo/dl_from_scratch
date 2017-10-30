import numpy as np

# def softmax(a):
#     exp_a =  np.exp(a)
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a / sum_exp_a
#
#     return y


# >>> a = np.array([1010, 1000, 900])
# >>> np.exp(a)
# __main__:1: RuntimeWarning: overflow encountered in exp
# array([ inf,  inf,  inf])
# >>> np.exp(a) / np.sum(np.exp(a))
# __main__:1: RuntimeWarning: invalid value encountered in true_divide
# array([ nan,  nan,  nan])
# >>>
# >>> c = np.max(a)
# >>> a - c
# array([   0,  -10, -110])
# >>>
# >>> np.exp(a -c) / np.sum(np.exp(a -c))
# array([  9.99954602e-01,   4.53978687e-05,   1.68883521e-48])

def  softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a


    return y

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)
print(np.sum(y))

