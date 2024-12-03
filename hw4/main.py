import cv2
import numpy as np
import os
import trimesh
import random
from icecream import ic

from config import load
from utils import feature_matching, ransac
from draw import draw_epipolar_lines, visulize_3d_points

THE_ANSWER_TO_LIFE_THE_UNIVERSE_AND_EVERYTHING = 42

Point = tuple[int, int]

random.seed(THE_ANSWER_TO_LIFE_THE_UNIVERSE_AND_EVERYTHING)
np.random.seed(THE_ANSWER_TO_LIFE_THE_UNIVERSE_AND_EVERYTHING)


def decompose_essential_matrix(E):
    """
    Decompose the essential matrix into rotation and translation components.

    Parameters:
    - E: Essential matrix.

    Returns:
    - R1, R2: Possible rotation matrices.
    - t: Translation vector.
    """
    # Singular value decomposition
    U, S, Vt = np.linalg.svd(E)
    m = (S[0] + S[1]) / 2
    E = U @ np.diag([m, m, 0]) @ Vt
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    U, S, Vt = np.linalg.svd(E)
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]

    # Ensure that the determinant of the rotation matrices is positive
    if np.linalg.det(R1) < 0:
        R1 *= -1
    if np.linalg.det(R2) < 0:
        R2 *= -1

    return R1, R2, t


def triangulate_3d_points(K, R, t, pts1, pts2):
    """
    Triangulates 3D points from two views given projection matrices and corresponding points.

    Parameters:
    - points1: Nx2 array of 2D points in the first image.
    - points2: Nx2 array of 2D points in the second image.
    - P1: 3x4 camera projection matrix for the first view.
    - P2: 3x4 camera projection matrix for the second view.

    Returns:
    - points_3d: Nx3 array of triangulated 3D points (inhomogeneous coordinates).
    """

    p1 = K[0] @ np.hstack((np.eye(3), np.zeros((3, 1))))
    p2 = K[1] @ np.hstack((R, t.reshape(-1, 1)))

    points_3d = []
    for i in range(len(pts1)):
        A = np.array(
            [
                pts1[i][0] * p1[2] - p1[0],
                pts1[i][1] * p1[2] - p1[1],
                pts2[i][0] * p2[2] - p2[0],
                pts2[i][1] * p2[2] - p2[1],
            ]
        )
        _, _, Vt = np.linalg.svd(A)
        point = Vt[-1, :3] / Vt[-1, 3]
        points_3d.append(point)

    neg = np.count_nonzero(np.array(points_3d)[:, 2] < 0)

    return np.array(points_3d), neg


def main(case: int):
    IMG_PATH, K, OUTPUT_PATH = load(case)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    print(f"Running images: {IMG_PATH}")
    print(f"Output path: {OUTPUT_PATH}")

    img1 = cv2.imread(IMG_PATH[0], cv2.IMREAD_COLOR)
    img2 = cv2.imread(IMG_PATH[1], cv2.IMREAD_COLOR)

    m1 = np.hstack((img1, img2))
    cv2.imwrite(os.path.join(OUTPUT_PATH, "original_image.jpg"), m1)

    matches: list[list[Point]] = feature_matching(img1, img2, 0.75, OUTPUT_PATH)
    ic(len(matches))
    assert len(matches[0]) == 2

    F = ransac(matches, threshold=5, max_iterations=800)
    ic(F)
    # Draw epipolar lines
    img1_line, img1_point, img2_line, img2_point = draw_epipolar_lines(
        img1, img2, np.array(matches)[:, 0], np.array(matches)[:, 1], F
    )
    cv2.imwrite(
        os.path.join(OUTPUT_PATH, "epipolar_lines.jpg"),
        np.vstack(
            (np.hstack((img1_line, img1_point)), np.hstack((img2_line, img2_point)))
        ),
    )

    E = K[1].T @ F @ K[0]

    R1_s, R2_s, t_s = decompose_essential_matrix(E)

    point3d = np.array([])
    negtive = float("inf")
    R_final, t_final = np.array([]), np.array([])

    ic(R1_s, R2_s, t_s)
    for R, t in [(R1_s, t_s), (R2_s, t_s), (R1_s, -t_s), (R2_s, -t_s)]:
        cur, cur_neg = triangulate_3d_points(
            K, R, t, np.array(matches)[:, 0], np.array(matches)[:, 1]
        )

        ic(R, t, cur_neg)
        if cur_neg < negtive:
            point3d, negtive = cur, cur_neg
            R_final, t_final = R, t

    ic(negtive, len(point3d), point3d)

    P1 = K[0] @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K[1] @ np.hstack((R_final, t_final.reshape(-1, 1)))

    # Save the triangulated points
    mesh = trimesh.Trimesh(vertices=point3d)
    mesh.export(os.path.join(OUTPUT_PATH, "output_model.obj"))

    img1_pts, img2_pts = visulize_3d_points(point3d, img1, img2, P1, P2, OUTPUT_PATH)
    cv2.imwrite(
        os.path.join(OUTPUT_PATH, "3d_points.jpg"), np.hstack((img1_pts, img2_pts))
    )


if __name__ == "__main__":
    main(1)
    main(2)
