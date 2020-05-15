# Copyright Ryan-Rhys Griffiths and Aditya Raymond Thawani 2020
# Author: Ryan-Rhys Griffiths
"""
Molecule kernels for Gaussian Process Regression implemented in GPflow.
"""

import gpflow
from gpflow.utilities import positive
from gpflow.utilities.ops import broadcasting_elementwise
import tensorflow as tf


class Tanimoto(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__()
        self.variance = gpflow.Parameter(1.0, transform=positive())

    def K(self, X, X2=None):
        """
        Compute the Tanimoto coefficient <x, y> / ||x||^2 + ||y||^2 - <x, y>

        :param X:
        :param X2:
        :return:
        """
        if X2 is None:
            X2 = X

        Xs = tf.reduce_sum(tf.square(X), axis=-1)
        X2s = tf.reduce_sum(tf.square(X2), axis=-1)
        cross_product = tf.tensordot(X, X2, [[-1], [-1]])
        denominator = -cross_product + broadcasting_elementwise(tf.add, Xs, X2s)

        return self.variance * cross_product/denominator  # this returns a 2D tensor

    def K_diag(self, X):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))
