import numpy as np
import scipy.optimize as spo

from impl.util import MakeHomogeneous, HNormalize


# Compute the reprojection error for a single correspondence
def ReprojectionError(P, point3D, point2D):
    # TODO --> DONE!
    # Project the 3D point into the image and compare it to the keypoint.
    # Make sure to properly normalize homogeneous coordinates.
    # point3D_norm = np.reshape(point3D, (3, 1))/np.linalg.norm(point3D)
    # err = np.vstack([np.reshape(point2D, (2, 1)), [1]]) - (P @ np.vstack([point3D_norm, [1]]))
    point3D_proj = P @ np.vstack([np.reshape(point3D, (3, 1)), [1]])
    point3D_proj_norm = point3D_proj / point3D_proj[-1][-1]
    err = np.vstack([np.reshape(point2D, (2, 1)), [1]]) - point3D_proj_norm
    err = [err[0, 0], err[1, 0]]
    return err  # TODO ---> DONE!


# Compute the residuals for all correspondences of the image
def ImageResiduals(P, points2D, points3D):
    num_residuals = points2D.shape[0]
    res = np.zeros(num_residuals * 2)

    for res_idx in range(num_residuals):
        p3D = points3D[res_idx]
        p2D = points2D[res_idx]

        err = ReprojectionError(P, p3D, p2D)

        res[res_idx * 2:res_idx * 2 + 2] = err

    # print('Reprojection error {}'.format(np.linalg.norm(res) ** 2))  # To observe the squared norm of the reprojection error at each minimization iteration

    return res


# Optimize the projection matrix given the 2D-3D point correspondences.
# 2D and 3D points with the same index are assumed to correspond.
def OptimizeProjectionMatrix(P, points2D, points3D):
    # The optimization requires a scalar cost value.
    # We use the sum of squared differences of all correspondences
    # print(P / P[2, 3])  # to check the projection matrix BEFORE optimization

    f = lambda x: np.linalg.norm(ImageResiduals(np.reshape(x, (3, 4)), points2D, points3D)) ** 2

    # Since the projection matrix is scale invariant we have an open degree of freedom from just the constraints.
    # Make sure this is fixed by keeping the last component close to 1.
    scale_constraint = {'type': 'eq', 'fun': lambda x: x[11] - 1}

    # Make sure the scale constraint is fulfilled at the beginning
    result = spo.minimize(f, np.reshape(P / P[2, 3], 12), options={'disp': True}, constraints=[scale_constraint], tol=1e-12)

    # print(np.reshape(result.x, (3, 4)))  # to check the projection matrix BEFORE optimization

    return np.reshape(result.x, (3, 4))
