import numpy as np
import cv2


def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin):
    frame = np.asarray(frame)
    patch = frame[ymin:ymax, xmin:xmax, :]
    # ----------------A)----------------------------------------------------------------------------------------------
    # hist, _ = np.histogramdd(patch, bins=(hist_bin, hist_bin, hist_bin))
    # ----------------B)----------------------------------------------------------------------------------------------
    # hist = cv2.calcHist([patch], [0, 1, 2], None, [hist_bin, hist_bin, hist_bin], [0, 256, 0, 256, 0, 256])
    # ----------------C)----------------------------------------------------------------------------------------------
    bin_step = 256./hist_bin
    hist = np.zeros((hist_bin, hist_bin, hist_bin))
    bins = (np.floor(patch/bin_step)).astype(int)
    binsR = bins[:, :, 0].flatten()
    binsG = bins[:, :, 1].flatten()
    binsB = bins[:, :, 2].flatten()
    hist[binsR, binsG, binsB] += 1

    hist = hist/(np.sum(hist) + 1e-9)
    return hist
