import numpy as np
from color_histogram import color_histogram
from chi2_cost import chi2_cost


def observe(particles, frame, bbox_height, bbox_width, hist_bin, hist, sigma):
    particles_w = np.zeros(particles.shape[0])
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    for i, particle in enumerate(particles):
        # -------------------A)-------------------------------------------------------------------
        # x, y = particle[0], particle[1]
        # half_w = bbox_width/2.
        # half_h = bbox_height/2.
        # xmin = np.max([x-half_w, 0])
        # xmax = np.min([x+half_w, frame_width-1])
        # ymin = np.max([y-half_h, 0])
        # ymax = np.min([y+half_h, frame_height-1])
        # new_hist = color_histogram(int(xmin), int(xmax), int(ymin), int(ymax), frame, hist_bin)
        # dist = chi2_cost(new_hist, hist)
        # -------------------B)-------------------------------------------------------------------
        new_hist = color_histogram(
            min(max(0, round(particles[i, 0]-0.5*bbox_width)), frame_width-1),
            min(max(0, round(particles[i, 1]-0.5*bbox_height)), frame_height-1),
            min(max(0, round(particles[i, 0]+0.5*bbox_width)), frame_width-1),
            min(max(0, round(particles[i, 1]+0.5*bbox_height)), frame_height-1),
            frame, hist_bin)
        dist = chi2_cost(new_hist, hist)

        particles_w[i] = (1./(np.sqrt(2.*np.pi)*sigma))*np.exp(-(dist**2)/(2.*(sigma**2)))

    particles_w = particles_w/np.sum(particles_w)
    return particles_w
