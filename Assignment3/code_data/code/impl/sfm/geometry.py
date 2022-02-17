import numpy as np

from impl.dlt import BuildProjectionConstraintMatrix
from impl.util import MakeHomogeneous, HNormalize
from impl.sfm.corrs import GetPairMatches
# from impl.opt import ImageResiduals, OptimizeProjectionMatrix

# # Debug
# import matplotlib.pyplot as plt
# from impl.vis import Plot3DPoints, PlotCamera, PlotProjectedPoints


def EstimateEssentialMatrix(K, im1, im2, matches):
    # TODO --> DONE!
    # Normalize coordinates (to points on the normalized image plane)
    K_inv = np.linalg.inv(K)

    # These are the keypoints on the normalized image plane (not to be confused with the normalization in the calibration exercise)
    normalized_kps1 = (K_inv @ MakeHomogeneous(im1.kps, 1).transpose()).transpose()
    normalized_kps2 = (K_inv @ MakeHomogeneous(im2.kps, 1).transpose()).transpose()

    # TODO --> DONE!
    # Assemble constraint matrix
    constraint_matrix = np.zeros((matches.shape[0], 9))

    for i in range(matches.shape[0]):
        # TODO --> DONE!
        # Add the constraints
        # constraint_matrix[i, :] = [normalized_kps1[matches[i, 0], 0] * normalized_kps2[matches[i, 1], 0],
        #                            normalized_kps1[matches[i, 0], 0] * normalized_kps2[matches[i, 1], 1],
        #                            normalized_kps1[matches[i, 0], 0],
        #                            normalized_kps1[matches[i, 0], 1] * normalized_kps2[matches[i, 1], 0],
        #                            normalized_kps1[matches[i, 0], 1] * normalized_kps2[matches[i, 1], 1],
        #                            normalized_kps1[matches[i, 0], 1],
        #                            normalized_kps2[matches[i, 1], 0], normalized_kps2[matches[i, 1], 1],
        #                            1]

        # 'Inverse' constraint matrix to recover E from kp2'*E*kp1 instead of kp1'*E*kp2
        constraint_matrix[i, :] = [normalized_kps2[matches[i, 1], 0] * normalized_kps1[matches[i, 0], 0],
                                   normalized_kps2[matches[i, 1], 0] * normalized_kps1[matches[i, 0], 1],
                                   normalized_kps2[matches[i, 1], 0],
                                   normalized_kps2[matches[i, 1], 1] * normalized_kps1[matches[i, 0], 0],
                                   normalized_kps2[matches[i, 1], 1] * normalized_kps1[matches[i, 0], 1],
                                   normalized_kps2[matches[i, 1], 1],
                                   normalized_kps1[matches[i, 0], 0], normalized_kps1[matches[i, 0], 1],
                                   1]

    # Solve for the nullspace of the constraint matrix
    _, _, vh = np.linalg.svd(constraint_matrix)
    vectorized_E_hat = vh[-1, :]

    # TODO --> DONE!
    # Reshape the vectorized matrix to its proper shape again
    E_hat = np.reshape(vectorized_E_hat, (3, 3), order='C')

    # TODO --> DONE!
    # We need to fulfill the internal constraints of E
    # The first two singular values need to be equal, the third one zero.
    # Since E is up to scale, we can choose the two equal singular values arbitrarily
    u_E, s_E, v_E_transp = np.linalg.svd(E_hat)

    s_E = np.eye(3)
    s_E[2, 2] = 0
    E = u_E @ s_E @ v_E_transp

    # This is just a quick test that should tell you if your estimated matrix is not correct
    # It might fail if you estimated E in the other direction (i.e. kp2' * E * kp1)
    # You can adapt it to your assumptions.
    for i in range(matches.shape[0]):
        kp1 = normalized_kps1[matches[i, 0], :]
        kp2 = normalized_kps2[matches[i, 1], :]

        # assert (abs(kp1.transpose() @ E @ kp2) < 0.01)
        assert (abs(kp2.transpose() @ E @ kp1) < 0.01)

    return E


def DecomposeEssentialMatrix(E):
    u, s, vh = np.linalg.svd(E)

    # Determine the translation up to sign
    t_hat = u[:, -1]

    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])

    # Compute the two possible rotations
    R1 = u @ W @ vh
    R2 = u @ W.transpose() @ vh

    # Make sure the orthogonal matrices are proper rotations
    if np.linalg.det(R1) < 0:
        R1 *= -1

    if np.linalg.det(R2) < 0:
        R2 *= -1

    # Assemble the four possible solutions
    sols = [
        (R1, t_hat),
        (R2, t_hat),
        (R1, -t_hat),
        (R2, -t_hat)
    ]

    return sols


def TriangulatePoints(K, im1, im2, matches):
    R1, t1 = im1.Pose()
    R2, t2 = im2.Pose()
    P1 = K @ np.append(R1, np.expand_dims(t1, 1), 1)
    P2 = K @ np.append(R2, np.expand_dims(t2, 1), 1)

    # Ignore matches that already have a triangulated point
    new_matches = np.zeros((0, 2), dtype=int)

    num_matches = matches.shape[0]
    for i in range(num_matches):
        p3d_idx1 = im1.GetPoint3DIdx(matches[i, 0])
        p3d_idx2 = im2.GetPoint3DIdx(matches[i, 1])
        if p3d_idx1 == -1 and p3d_idx2 == -1:
            new_matches = np.append(new_matches, matches[[i]], 0)

    num_new_matches = new_matches.shape[0]

    points3D = np.zeros((num_new_matches, 3))

    for i in range(num_new_matches):
        kp1 = im1.kps[new_matches[i, 0], :]
        kp2 = im2.kps[new_matches[i, 1], :]

        # H & Z Sec. 12.2
        A = np.array([
            kp1[0] * P1[2] - P1[0],
            kp1[1] * P1[2] - P1[1],
            kp2[0] * P2[2] - P2[0],
            kp2[1] * P2[2] - P2[1]
        ])

        _, _, vh = np.linalg.svd(A)
        homogeneous_point = vh[-1]
        points3D[i] = homogeneous_point[:-1] / homogeneous_point[-1]

    # We need to keep track of the correspondences between image points and 3D points
    im1_corrs = new_matches[:, 0]
    im2_corrs = new_matches[:, 1]

    # TODO --> DONE!
    # Filter points behind the cameras by transforming them into each camera space and checking the depth (Z)
    # Make sure to also remove the corresponding rows in `im1_corrs` and `im2_corrs`
    points3D_im1 = ((R1 @ points3D.transpose()) + np.expand_dims(t1, 1)).transpose()
    points3D_im2 = ((R2 @ points3D.transpose()) + np.expand_dims(t2, 1)).transpose()

    points3D_cleaned = points3D*np.expand_dims(points3D_im1[:, -1] > 0.0, 1)*np.expand_dims(points3D_im2[:, -1] > 0.0, 1)
    points3D = points3D[~np.all(points3D_cleaned == 0, axis=1)]
    im1_corrs = im1_corrs[~np.all(points3D_cleaned == 0, axis=1)]
    im2_corrs = im2_corrs[~np.all(points3D_cleaned == 0, axis=1)]

    return points3D, im1_corrs, im2_corrs


def EstimateImagePose(points2D, points3D, K):
    # TODO --> DONE!
    # We use points in the normalized image plane.
    # This removes the 'K' factor from the projection matrix.
    # We don't normalize the 3D points here to keep the code simpler.
    K_inv = np.linalg.inv(K)
    normalized_points2D = (K_inv @ MakeHomogeneous(points2D, 1).transpose()).transpose()

    constraint_matrix = BuildProjectionConstraintMatrix(normalized_points2D, points3D)

    # We don't use optimization here since we would need to make sure to only optimize on the se(3) manifold
    # (the manifold of proper 3D poses). This is a bit too complicated right now.
    # Just DLT should give good enough results for this dataset.

    # Solve for the nullspace
    _, _, vh = np.linalg.svd(constraint_matrix)
    P_vec = vh[-1, :]
    P = np.reshape(P_vec, (3, 4), order='C')

    # Make sure we have a proper rotation
    u, s, vh = np.linalg.svd(P[:, :3])
    R = u @ vh

    if np.linalg.det(R) < 0:
        R *= -1

    _, _, vh = np.linalg.svd(P)
    C = np.copy(vh[-1, :])

    t = -R @ (C[:3] / C[3])

    return R, t


def TriangulateImage(K, image_name, images, registered_images, matches):
    # TODO --> DONE!
    # Loop over all registered images and triangulate new points with the new image.
    # Make sure to keep track of all new 2D-3D correspondences, also for the registered images

    image = images[image_name]
    points3D = np.empty((0, 3))
    corrs = {}

    # You can save the correspondences for each image in a dict and refer to the `local` new point indices here.
    # Afterwards you just add the index offset before adding the correspondences to the images.
    for reg_image_name in registered_images:
        offset = len(points3D)
        points3D_new, im1_corrs, im2_corrs = TriangulatePoints(K, image, images[reg_image_name], GetPairMatches(image_name, reg_image_name, matches))
        points3D = np.append(points3D, points3D_new, axis=0)
        corrs[image_name] = np.array([im1_corrs, offset+np.arange(points3D_new.shape[0])])
        corrs[reg_image_name] = np.array([im2_corrs, offset+np.arange(points3D_new.shape[0])])

    return points3D, corrs
