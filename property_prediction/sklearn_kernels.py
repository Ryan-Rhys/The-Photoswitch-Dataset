# Author: Dries Van Rompaey
"""
Implementation of the Tanimoto kernel in sklearn.
"""

import numpy as np
from sklearn.gaussian_process.kernels import StationaryKernelMixin, NormalizedKernelMixin,Kernel, Hyperparameter


class TanimotoKernel(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """
    Based on https://github.com/Ryan-Rhys/The-Photoswitch-Dataset/blob/master/examples/gp_regression_on_molecules.ipynb

    Tanimoto kernel.
        Note: the kernel can occasionally suffer from convergence issues.
        If this is the case, try increasing the gaussianprocess alpha parameter slightly.
        Arguments:
        - variance: variance of your kernel (float, default: 1.0)
        - variance_bounds: optimization bounds of the variance (tuple of floats or fixed, default: fixed)
    """
    def __init__(self, variance=1.0, variance_bounds='fixed'):
        self.variance = variance
        self.variance_bounds = variance_bounds

    @property
    def hyperparameter_variance(self):
        return Hyperparameter(
            "variance", "numeric", self.variance_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        X1s = np.sum(np.square(X), axis=-1)  # Squared L2-norm of X
        X2s = np.sum(np.square(Y), axis=-1)  # Squared L2-norm of X
        outer_product = np.tensordot(X, Y, axes=([-1], [-1]))  # outer product of the matrices X and X2
        denominator = -outer_product + (X1s[:, np.newaxis] + X2s)
        out = self.variance * outer_product/denominator
        if eval_gradient:
            if not self.hyperparameter_variance.fixed:
                return out, (outer_product/denominator)[:, :, np.newaxis]
            else:
                return out, np.empty((X.shape[0], X.shape[0], 0))
        return out
