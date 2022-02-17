import numpy as np


def resample(particles, particles_w):
    possible_idxs = np.arange(particles.shape[0])
    idxs = np.random.choice(possible_idxs, size=particles.shape[0], p=particles_w)
    particles_updated = particles[idxs, :]
    particles_w_updated = particles_w[idxs]/np.sum(particles_w[idxs])
    return particles_updated, particles_w_updated
