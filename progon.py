import numpy as np
def progonka(a, c, b, d):
    n = len(c)
    alpha = np.zeros(n)
    beta = np.zeros(n)
    alpha[0] = -b[0] / c[0]
    beta[0] = d[0] / c[0]
    for i in range(1, n):
        denom = c[i] + a[i-1] * alpha[i-1]
        if i < n - 1:
            alpha[i] = -b[i] / denom
        beta[i] = (d[i] - a[i-1] * beta[i-1]) / denom
    x = np.zeros(n)
    x[-1] = beta[-1]
    for I in reversed(range(n - 1)):
        x[i] = alpha[i] * x[I + 1] + beta[i]
    return x
