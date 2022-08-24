"""
@author: Yong Zheng Ong

code to generate ellipses dataset.

requires odl to generate, which can be installed from https://github.com/odlgroup/odl

with references from 
https://github.com/adler-j/learned_gradient_tomography
https://github.com/odlgroup/odl
"""
import os

import numpy as np
import odl

ntrain = 10000
ntest = 1000

def random_ellipse():
    return ((np.random.rand() - 0.3) * np.random.exponential(0.3),
            np.random.exponential() * 0.2, np.random.exponential() * 0.2,
            np.random.rand() - 0.5, np.random.rand() - 0.5,
            np.random.rand() * 2 * np.pi)

def random_phantom(spc):
    n = np.random.poisson(100)
    ellipses = [random_ellipse() for _ in range(n)]
    return odl.phantom.ellipsoid_phantom(spc, ellipses)

size = 128
space = odl.uniform_discr([-64, -64], [64, 64], [size, size],
                          dtype='float32')

geometry = odl.tomo.parallel_beam_geometry(space, num_angles=30)
operator = odl.tomo.RayTransform(space, geometry)
pseudoinverse = odl.tomo.fbp_op(operator)

# Ensure operator has fixed operator norm for scale invariance
opnorm = odl.power_method_opnorm(operator)
operator = (1 / opnorm) * operator
pseudoinverse = pseudoinverse * opnorm

def generate_data(ntrain):
    """Generate a set of random data."""
    n_iter = ntrain

    x_arr = np.empty((n_iter, space.shape[0], space.shape[1], 1), dtype='float32')
    y_arr = np.empty((n_iter, operator.range.shape[0], operator.range.shape[1], 1), dtype='float32')
    x_true_arr = np.empty((n_iter, space.shape[0], space.shape[1], 1), dtype='float32')
    y_true_arr = np.empty((n_iter, operator.range.shape[0], operator.range.shape[1], 1), dtype='float32')

    for i in range(n_iter):
        phantom = random_phantom(space)
        data = operator(phantom)
        noisy_data = data + odl.phantom.white_noise(operator.range) * np.mean(np.abs(data)) * 0.05
        fbp = pseudoinverse(noisy_data)

        x_arr[i, ..., 0] = fbp # 128, 128
        x_true_arr[i, ..., 0] = phantom # 128, 128
        y_arr[i, ..., 0] = noisy_data # 30, 128
        y_true_arr[i, ..., 0] = noisy_data # 30, 128

    return x_arr, y_arr, x_true_arr, y_true_arr

x_arr, y_arr, x_true_arr, y_true_arr = generate_data(ntrain)
np.savez("./data/train_data.npz", x_arr, y_arr, x_true_arr, y_true_arr)
x_arr, y_arr, x_true_arr, y_true_arr= generate_data(ntest)
np.savez("./data/test_data_ellipses.npz", x_arr, y_arr, x_true_arr, y_true_arr)
x_arr, y_arr, x_true_arr, y_true_arr= generate_data(0, validation=True)
np.savez("./data/test_data_phantom.npz", x_arr, y_arr, x_true_arr, y_true_arr)
