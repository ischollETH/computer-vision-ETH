import numpy as np
import cv2
import glob
import os
from sklearn.cluster import KMeans
from tqdm import tqdm


def findnn(D1, D2):
    """
    :param D1: NxD matrix containing N feature vectors of dim. D
    :param D2: MxD matrix containing M feature vectors of dim. D
    :return:
        Idx: N-dim. vector containing for each feature vector in D1 the index of the closest feature vector in D2.
        Dist: N-dim. vector containing for each feature vector in D1 the distance to the closest feature vector in D2
    """
    N = D1.shape[0]
    M = D2.shape[0]  # [k]

    # Find for each feature vector in D1 the nearest neighbor in D2
    Idx, Dist = [], []
    for i in range(N):
        minidx = 0
        mindist = np.linalg.norm(D1[i, :] - D2[0, :])
        for j in range(1, M):
            d = np.linalg.norm(D1[i, :] - D2[j, :])

            if d < mindist:
                mindist = d
                minidx = j
        Idx.append(minidx)
        Dist.append(mindist)
    return Idx, Dist


def grid_points(img, nPointsX, nPointsY, border):
    """
    :param img: input gray img, numpy array, [h, w]
    :param nPointsX: number of grids in x dimension
    :param nPointsY: number of grids in y dimension
    :param border: leave border pixels in each image dimension
    :return: vPoints: 2D grid point coordinates, numpy array, [nPointsX*nPointsY, 2]
    """
    # todo --> DONE!
    h, w = img.shape
    x = np.linspace(border, w-border, nPointsX, dtype=int)
    y = np.linspace(border, h-border, nPointsY, dtype=int)
    x_grid, y_grid = np.meshgrid(x, y)
    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()
    vPoints = np.array(np.hstack((x_grid[..., None], y_grid[..., None]))) # numpy array, [nPointsX*nPointsY, 2]

    return vPoints


def descriptors_hog(img, vPoints, cellWidth, cellHeight):
    nBins = 8
    w = cellWidth
    h = cellHeight

    grad_x = cv2.Sobel(img, cv2.CV_16S, dx=1, dy=0, ksize=1)
    grad_y = cv2.Sobel(img, cv2.CV_16S, dx=0, dy=1, ksize=1)

    mag, angle = cv2.cartToPolar(grad_x.astype(float), grad_y.astype(float), angleInDegrees=True)
    mag = np.asarray(mag)

    # -------------------------When using angles 0-180°--------------------------------------------------------------
    angle = np.asarray(abs((angle > 180)*360 - angle))
    step180 = 180/nBins
    angle_div = angle/step180
    # -------------------------When using angles 0-360°-------------------------------------------------------------
    # step360 = 360/nBins
    # angle_div = np.asarray(angle)/step360

    angle_floor = np.floor(angle_div).astype(int)
    angle_floor[angle_floor == nBins] = 0
    angle_proportion = angle_div - angle_floor
    angle_ceil = np.ceil(angle_div).astype(int)
    angle_ceil[angle_ceil == nBins] = 0

    descriptors = []  # list of descriptors for the current image, each entry is one 128-d vector for a grid point
    for i in range(len(vPoints)):
        # todo --> DONE!
        x = vPoints[i, 0]
        y = vPoints[i, 1]
        single_histogram = []
        for j in range(-2*h, 2*h, 4):
            for l in range(-2*w, 2*w, 4):
                single_bin = np.zeros(nBins)
                for s in range(j, j+(h-1), 1):
                    for t in range(l, l+(w-1), 1):
                        # ------------------A) ignoring magnitude & proportion---------------------------------
                        # single_bin[angle_floor[y+s, x+t]] += 1
                        # ------------------B) using magnitude & ignoring proportion---------------------------
                        # single_bin[angle_floor[y+s, x+t]] += mag[y+s, x+t]
                        # ------------------C) ignoring magnitude & using proportion---------------------------
                        # single_bin[angle_floor[y+s, x+t]] += angle_proportion[y+s, x+t]
                        # single_bin[angle_ceil[y+s, x+t]] += (1-angle_proportion[y+s, x+t])
                        # ------------------D) using magnitude & using proportion------------------------------
                        single_bin[angle_floor[y+s, x+t]] += angle_proportion[y+s, x+t]*mag[y+s, x+t]
                        single_bin[angle_ceil[y+s, x+t]] += (1-angle_proportion[y+s, x+t])*mag[y+s, x+t]
                single_histogram = np.hstack((single_histogram, single_bin))
        descriptors.append(single_histogram)

    descriptors = np.asarray(descriptors)  # [nPointsX*nPointsY, 128], descriptor for the current image (100 grid points)
    return descriptors


def create_codebook(nameDirPos, nameDirNeg, k, numiter):
    """
    :param nameDirPos: dir to positive training images
    :param nameDirNeg: dir to negative training images
    :param k: number of kmeans cluster centers
    :param numiter: maximum iteration numbers for kmeans clustering
    :return: vCenters: center of kmeans clusters, numpy array, [k, 128]
    """
    vImgNames = sorted(glob.glob(os.path.join(nameDirPos, '*.png')))
    vImgNames = vImgNames + sorted(glob.glob(os.path.join(nameDirNeg, '*.png')))

    nImgs = len(vImgNames)

    cellWidth = 4
    cellHeight = 4
    nPointsX = 10
    nPointsY = 10
    border = 8

    vFeatures = []  # list for all features of all images (each feature: 128-d, 16 histograms containing 8 bins)
    # Extract features for all image
    for i in tqdm(range(nImgs)):
        # print('processing image {} ...'.format(i+1))
        img = cv2.imread(vImgNames[i])  # [172, 208, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # [h, w]

        # Collect local feature points for each image, and compute a descriptor for each local feature point
        # todo --> DONE!
        vPoints = grid_points(img, nPointsX, nPointsY, border)
        descriptors = descriptors_hog(img, vPoints, cellWidth, cellHeight)
        vFeatures.append(descriptors)

    vFeatures = np.asarray(vFeatures)  # [n_imgs, n_vPoints, 128]
    vFeatures = vFeatures.reshape(-1, vFeatures.shape[-1])  # [n_imgs*n_vPoints, 128]
    print('number of extracted features: ', len(vFeatures))

    # Cluster the features using K-Means
    print('clustering ...')
    kmeans_res = KMeans(n_clusters=k, max_iter=numiter).fit(vFeatures)
    vCenters = kmeans_res.cluster_centers_  # [k, 128]
    print('done clustering')
    return vCenters


def bow_histogram(vFeatures, vCenters):
    """
    :param vFeatures: MxD matrix containing M feature vectors of dim. D
    :param vCenters: NxD matrix containing N cluster centers of dim. D
    :return: histo: N-dim. numpy vector containing the resulting BoW activation histogram.
    """
    # todo --> DONE!
    N, D = vCenters.shape
    histo = np.zeros(N)
    for feature in vFeatures:
        dists = np.subtract(vCenters, feature)
        dists = np.linalg.norm(dists, axis=1)
        idx = np.argmin(dists)
        histo[idx] += 1

    return histo


def create_bow_histograms(nameDir, vCenters):
    """
    :param nameDir: dir of input images
    :param vCenters: kmeans cluster centers, [k, 128] (k is the number of cluster centers)
    :return: vBoW: matrix, [n_imgs, k]
    """
    vImgNames = sorted(glob.glob(os.path.join(nameDir, '*.png')))
    nImgs = len(vImgNames)

    cellWidth = 4
    cellHeight = 4
    nPointsX = 10
    nPointsY = 10
    border = 8

    # Extract features for all images in the given directory
    vBoW = []
    for i in tqdm(range(nImgs)):
        # print('processing image {} ...'.format(i + 1))
        img = cv2.imread(vImgNames[i])  # [172, 208, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # [h, w]

        # todo --> DONE!
        vPoints = grid_points(img, nPointsX, nPointsY, border)
        descriptors = descriptors_hog(img, vPoints, cellWidth, cellHeight)
        vBoW.append(bow_histogram(descriptors, vCenters))

    vBoW = np.asarray(vBoW)  # [n_imgs, k]
    return vBoW


def bow_recognition_nearest(histogram, vBoWPos, vBoWNeg):
    """
    :param histogram: bag-of-words histogram of a test image, [1, k]
    :param vBoWPos: bag-of-words histograms of positive training images, [n_imgs, k]
    :param vBoWNeg: bag-of-words histograms of negative training images, [n_imgs, k]
    :return: sLabel: predicted result of the test image, 0(without car)/1(with car)
    """

    # Find the nearest neighbor in the positive and negative sets and decide based on this neighbor
    # todo --> DONE!
    pos_dists = np.subtract(vBoWPos, histogram)
    pos_dists = np.linalg.norm(pos_dists, axis=1)
    pos_idx = np.argmin(pos_dists)
    DistPos = pos_dists[pos_idx]

    neg_dists = np.subtract(vBoWNeg, histogram)
    neg_dists = np.linalg.norm(neg_dists, axis=1)
    neg_idx = np.argmin(neg_dists)
    DistNeg = neg_dists[neg_idx]

    if (DistPos < DistNeg):
        sLabel = 1
    else:
        sLabel = 0
    return sLabel


if __name__ == '__main__':
    nameDirPos_train = 'data/data_bow/cars-training-pos'
    nameDirNeg_train = 'data/data_bow/cars-training-neg'
    nameDirPos_test = 'data/data_bow/cars-testing-pos'
    nameDirNeg_test = 'data/data_bow/cars-testing-neg'

    k = 50  # todo --> DONE!
    numiter = 300  # todo --> DONE!

    print('creating codebook ...')
    vCenters = create_codebook(nameDirPos_train, nameDirNeg_train, k, numiter)

    print('creating bow histograms (pos) ...')
    vBoWPos = create_bow_histograms(nameDirPos_train, vCenters)
    print('creating bow histograms (neg) ...')
    vBoWNeg = create_bow_histograms(nameDirNeg_train, vCenters)

    # test pos samples
    print('creating bow histograms for test set (pos) ...')
    vBoWPos_test = create_bow_histograms(nameDirPos_test, vCenters)  # [n_imgs, k]
    result_pos = 0
    print('testing pos samples ...')
    for i in range(vBoWPos_test.shape[0]):
        cur_label = bow_recognition_nearest(vBoWPos_test[i:(i + 1)], vBoWPos, vBoWNeg)
        result_pos = result_pos + cur_label
    acc_pos = result_pos / vBoWPos_test.shape[0]
    print('test pos sample accuracy:', acc_pos)

    # test neg samples
    print('creating bow histograms for test set (neg) ...')
    vBoWNeg_test = create_bow_histograms(nameDirNeg_test, vCenters)  # [n_imgs, k]
    result_neg = 0
    print('testing neg samples ...')
    for i in range(vBoWNeg_test.shape[0]):
        cur_label = bow_recognition_nearest(vBoWNeg_test[i:(i + 1)], vBoWPos, vBoWNeg)
        result_neg = result_neg + cur_label
    acc_neg = 1 - result_neg / vBoWNeg_test.shape[0]
    print('test neg sample accuracy:', acc_neg)
