import time
import os
import random
import math
import torch
import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
import skimage
from skimage import io, color
from skimage.transform import rescale


def distance(x, X):
    # t1 = time.time()
    dist = torch.norm(X-x, dim=1)
    # dist = torch.zeros((X.shape[0], 1))
    # for i, element in enumerate(X):
    #     dist[i] = torch.norm(element-x)
    # t2 = time.time() - t1
    # print('dist took {}s'.format(t2))
    return dist


def distance_batch(x, X):
    dist = torch.cdist(x, X, p=2)
    return dist


def gaussian(dist, bandwidth):
    # t1 = time.time()
    # weights = (1/math.sqrt(2*math.pi*(bandwidth**2)))*torch.exp(-(dist**2)/(2*(bandwidth**2)))
    weights = torch.exp(-(dist**2)/(2*(bandwidth**2)))
    # t2 = time.time() - t1
    # print('gaussian took {}s'.format(t2))
    return weights


def update_point(weight, X):
    # t1 = time.time()
    weight = weight.reshape(len(weight), 1)
    weighted_X = torch.mul(weight, X)
    update_value = torch.sum(weighted_X, 0)/torch.sum(weight, 0)
    # t2 = time.time() - t1
    # print('update_point took {}s'.format(t2))
    return update_value


def update_point_batch(weight, X):
    weighted_X = torch.matmul(weight, X)
    weight_sum = torch.sum(weight, 1)
    weight_sum = weight_sum.reshape(len(weight_sum), 1)
    update_value = weighted_X/weight_sum
    return update_value


def meanshift_step(X, bandwidth=2.5):
    X_ = X.clone()
    for i, x in enumerate(X):
        dist = distance(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i] = update_point(weight, X)
    return X_


def meanshift_step_batch(X, bandwidth=2.5):
    X_ = X.clone()
    batch_size = 55
    dist = torch.zeros(batch_size, X.shape[0]).double()
    for i in range(0, len(X), batch_size):
        x = X[i:i+batch_size]
        dist = distance_batch(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i:i+batch_size] = update_point_batch(weight, X)
    return X_


def meanshift(X):
    X = X.clone()
    for _ in range(20):
        # X = meanshift_step(X)   # slow implementation
        X = meanshift_step_batch(X)   # fast implementation
    return X


scale = 0.25    # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('cow.jpg'), scale, multichannel=True)
image_lab = color.rgb2lab(image)

shape = image_lab.shape  # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Run your mean-shift algorithm
t = time.time()
X = meanshift(torch.from_numpy(image_lab)).detach().cpu().numpy()
# X = meanshift(torch.from_numpy(data).cuda()).detach().cpu().numpy()  # you can use GPU if you have one
t = time.time() - t
print('Elapsed time for mean-shift: {}'.format(t))

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, multichannel=True)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave('result.png', result_image)
