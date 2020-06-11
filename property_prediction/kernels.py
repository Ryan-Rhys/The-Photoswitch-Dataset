# Copyright Ryan-Rhys Griffiths and Aditya Raymond Thawani 2020
# Author: Ryan-Rhys Griffiths
"""
Molecule kernels for Gaussian Process Regression implemented in GPflow.
"""

import numpy as np
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
        Compute the Tanimoto kernel matrix σ² * ((<x, y>) / (||x||^2 + ||y||^2 - <x, y>))

        :param X: N x D array
        :param X2: M x D array. If None, compute the N x N kernel matrix for X.
        :return: The kernel matrix of dimension N x M
        """
        if X2 is None:
            X2 = X

        Xs = tf.reduce_sum(tf.square(X), axis=-1)  # Squared L2-norm of X
        X2s = tf.reduce_sum(tf.square(X2), axis=-1)  # Squared L2-norm of X2
        cross_product = tf.tensordot(X, X2, [[-1], [-1]])  # outer product of the matrices X and X2

        # Analogue of denominator in Tanimoto formula

        denominator = -cross_product + broadcasting_elementwise(tf.add, Xs, X2s)

        return self.variance * cross_product/denominator

    def K_diag(self, X):
        """
        Compute the diagonal of the N x N kernel matrix of X
        :param X: N x D array
        :return: N x 1 array
        """
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))

class Soap(gpflow.kernels.Kernel):

    def __init__(self, K_mat):
        super().__init__(active_dims=[0])
        self.var = gpflow.Parameter(10.0, transform=positive())
        self.mag = gpflow.Parameter(1.0, transform=positive())
        self.max_mag = K_mat.max()
        self.K_mat = tf.constant(K_mat, dtype=tf.float64)
        self.diag = tf.constant(np.diag(K_mat), dtype=tf.float64)
 
    def K(self, X, X2=None):
         """
         Compute the Soap Matern-32 kernel matrix
 
         :param X: N x D array
         :param X2: M x D array. If None, compute the N x N kernel matrix for X.
         :return: The kernel matrix of dimension N x M
         """
         if X2 is None:
            X2=X

         A = tf.cast(X,tf.int32)
         A = tf.reshape(A,[-1])
         A2 = tf.reshape(X2,[-1])
         A2 = tf.cast(A2,tf.int32)
         K_mat = tf.gather(self.K_mat, A, axis=0)
         K_mat = tf.gather(K_mat, A2, axis=1)
         z = tf.math.sqrt(6*(self.max_mag-K_mat))*self.var
         K_final = self.mag*(1+z)*tf.math.exp(-z)
         return K_final
 
    def K_diag(self, X, presliced=None):
        """
        Compute the diagonal of the N x N kernel matrix of X
        :param X: N x D array
        :return: N x 1 array
        """
        A=tf.cast(X,tf.int32)
        K_diag = tf.gather_nd(self.diag, A)
        z = tf.math.sqrt(6*(self.max_mag-K_diag))*self.var
        return self.mag*(1+z)*tf.math.exp(-z)
