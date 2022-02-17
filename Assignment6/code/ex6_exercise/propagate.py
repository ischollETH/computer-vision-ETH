import numpy as np


def propagate(particles, frame_height, frame_width, params):
    if params["model"] == 0:
        A = np.identity(2)
    else:
        A = np.identity(4)
        A[0, 2] = 1
        A[1, 3] = 1

    sigma_pos = params["sigma_position"]
    sigma_vel = params["sigma_velocity"]

    w = np.random.normal(loc=0, scale=sigma_pos, size=(2, particles.shape[0]))
    if params["model"] == 1:
        w_vel = np.random.normal(loc=0, scale=sigma_vel, size=(2, particles.shape[0]))
        w = np.vstack((w, w_vel))
    particles_prop = np.transpose(np.matmul(A, np.transpose(particles)) + w)

    particles_prop[:, 0] = np.minimum(particles_prop[:, 0], frame_width-1)
    particles_prop[:, 1] = np.minimum(particles_prop[:, 1], frame_height-1)
    particles_prop[:, 0:2] = np.maximum(particles_prop[:, 0:2], 0)

    return particles_prop

