import numpy as np


def estimate(particles, particles_w):
    mean_state = np.transpose(np.matmul(np.transpose(particles), particles_w))
    return mean_state
