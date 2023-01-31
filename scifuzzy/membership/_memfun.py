import numpy as np 

__all__ = ['dsigmf','gauss2mf','gaussmf','gbellmf','piecemf','pimf','psigmf','sigmf','smf','trapmf','trimf','zmf']

#Difference of two fuzzy sigmoid membership functions.

def dsigmf(x, b1, c1, b2, c2):
    return sigmf(x, b1, c=c1) - sigmf(x, b2, c=c2)



# Gaussian fuzzy membership function of two combined Gaussians.
def gauss2mf(x, mean1, sigma1, mean2, sigma2):
    assert mean1 <= mean2, 'mean1 <= mean2 is required.  See docstring.'
    y = np.ones(len(x))
    idx1 = x <= mean1
    idx2 = x > mean2
    y[idx1] = gaussmf(x[idx1], mean1, sigma1)
    y[idx2] = gaussmf(x[idx2], mean2, sigma2)
    return y

# Gaussian fuzzy membership function.
def gaussmf(x, mean, sigma):
    return np.exp(-((x - mean) ** 2.) / float(sigma) ** 2.)

#  Generalized Bell function fuzzy membership generator
def gbellmf(x, a, b, c):
    return 1. / (1. + np.abs((x - c) / a) ** (2 * b))

#  Piecewise linear membership function (particularly used in FIRE filters).
def piecemf(x, abc):
    pass
   

    



# Pi-function fuzzy membership generator.
def pimf(x, a, b, c, d):
    y = np.ones(len(x))
    assert a <= b and b <= c and c <= d, 'a <= b <= c <= d is required.'

    idx = x <= a
    y[idx] = 0

    idx = np.logical_and(a <= x, x <= (a + b) / 2.)
    y[idx] = 2. * ((x[idx] - a) / (b - a)) ** 2.

    idx = np.logical_and((a + b) / 2. < x, x <= b)
    y[idx] = 1 - 2. * ((x[idx] - b) / (b - a)) ** 2.

    idx = np.logical_and(c <= x, x < (c + d) / 2.)
    y[idx] = 1 - 2. * ((x[idx] - c) / (d - c)) ** 2.

    idx = np.logical_and((c + d) / 2. <= x, x <= d)
    y[idx] = 2. * ((x[idx] - d) / (d - c)) ** 2.

    idx = x >= d
    y[idx] = 0

    return y


# Product of two sigmoid membership functions.
def psigmf(x, b1, c1, b2, c2):
    return sigmf(x, b1, c1) * sigmf(x, b2, c2)

# The basic sigmoid membership function generator.
def sigmf(x, b, c):
    return 1. / (1. + np.exp(- c * (x - b)))

#  S-function fuzzy membership generator.
def smf(x, a, b):
    assert a <= b, 'a <= b is required.'
    y = np.ones(len(x))
    idx = x <= a
    y[idx] = 0

    idx = np.logical_and(a <= x, x <= (a + b) / 2.)
    y[idx] = 2. * ((x[idx] - a) / (b - a)) ** 2.

    idx = np.logical_and((a + b) / 2. <= x, x <= b)
    y[idx] = 1 - 2. * ((x[idx] - b) / (b - a)) ** 2.

    return y

# Trapezoidal membership function generator.
def trapmf(x, abcd):
    assert len(abcd) == 4, 'abcd parameter must have exactly four elements.'
    a, b, c, d = np.r_[abcd]
    assert a <= b and b <= c and c <= d, 'abcd requires the four elements \
                                          a <= b <= c <= d.'
    y = np.ones(len(x))

    idx = np.nonzero(x <= b)[0]
    y[idx] = trimf(x[idx], np.r_[a, b, b])

    idx = np.nonzero(x >= c)[0]
    y[idx] = trimf(x[idx], np.r_[c, c, d])

    idx = np.nonzero(x < a)[0]
    y[idx] = np.zeros(len(idx))

    idx = np.nonzero(x > d)[0]
    y[idx] = np.zeros(len(idx))

    return y


#  Triangular membership function generator.
def trimf(x, abc):
    assert len(abc) == 3, 'abc parameter must have exactly three elements.'
    a, b, c = np.r_[abc]     # Zero-indexing in Python
    assert a <= b and b <= c, 'abc requires the three elements a <= b <= c.'

    y = np.zeros(len(x))

    # Left side
    if a != b:
        idx = np.nonzero(np.logical_and(a < x, x < b))[0]
        y[idx] = (x[idx] - a) / float(b - a)

    # Right side
    if b != c:
        idx = np.nonzero(np.logical_and(b < x, x < c))[0]
        y[idx] = (c - x[idx]) / float(c - b)

    idx = np.nonzero(x == b)
    y[idx] = 1
    return y


# Z-function fuzzy membership generator.
def zmf(x, a, b):
    assert a <= b, 'a <= b is required.'

    y = np.ones(len(x))

    idx = np.logical_and(a <= x, x < (a + b) / 2.)
    y[idx] = 1 - 2. * ((x[idx] - a) / (b - a)) ** 2.

    idx = np.logical_and((a + b) / 2. <= x, x <= b)
    y[idx] = 2. * ((x[idx] - b) / (b - a)) ** 2.

    idx = x >= b
    y[idx] = 0

    return y


