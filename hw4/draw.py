import cv2
import os
import numpy as np

from random import randint, seed
from matplotlib import pyplot as plt


def random_color():
    # return (0, 255, 0)
    return tuple(randint(0, 255) for _ in range(3))


def draw_epipolar_lines(img1, img2, points1, points2, F):
    """
    Draws epipolar lines on two images given corresponding points and the fundamental matrix.

    Parameters:
    - img1, img2: Images where lines will be drawn (grayscale or color).
    - points1, points2: Corresponding points in the two images (Nx2 numpy arrays).
    - F: Fundamental matrix.

    Returns:
    - img1_lines, img2_lines: Images with epipolar lines drawn.
    """
    # Convert points to homogeneous format
    points1 = cv2.convertPointsToHomogeneous(points1).reshape(-1, 3)
    points2 = cv2.convertPointsToHomogeneous(points2).reshape(-1, 3)

    # Compute epipolar lines for points in the second image corresponding to points1
    lines_in_img2 = cv2.computeCorrespondEpilines(points1[:, :2], 1, F).reshape(-1, 3)

    # Compute epipolar lines for points in the first image corresponding to points2
    lines_in_img1 = cv2.computeCorrespondEpilines(points2[:, :2], 2, F).reshape(-1, 3)

    # Draw lines on the images
    img1_line = img1.copy()
    img1_point = img1.copy()
    img2_line = img2.copy()
    img2_point = img2.copy()

    def draw_line_point(img, lines, points, draw_line=True, draw_point=True):
        """Helper function to draw lines and points."""
        h, w, _ = img.shape
        for line, point in zip(lines, points):
            # Extract line parameters
            a, b, c = line
            # Find points to draw the line
            x0, y0 = 0, int(-c / b)
            x1, y1 = w, int(-(c + a * w) / b)
            color = random_color()
            # Draw the line
            if draw_line:
                cv2.line(img, (x0, y0), (x1, y1), color, 1)
            # Draw the point
            if draw_point:
                cv2.circle(img, tuple(map(int, point)), 5, (0, 255, 0), -1)

    # Draw epipolar lines and points
    draw_line_point(
        img1_line, lines_in_img1, points1[:, :2], draw_line=True, draw_point=False
    )
    draw_line_point(
        img1_point, lines_in_img1, points1[:, :2], draw_line=False, draw_point=True
    )
    draw_line_point(
        img2_line, lines_in_img2, points2[:, :2], draw_line=False, draw_point=True
    )
    draw_line_point(
        img2_point, lines_in_img2, points2[:, :2], draw_line=True, draw_point=False
    )

    return img1_line, img1_point, img2_line, img2_point


def visulize_3d_points(points_3d, img1, img2, P1, P2):
    """
    Visualize 3D points in two views given projection matrices and corresponding points.

    Parameters:
    - points_3d: Nx3 array of 3D points (inhomogeneous coordinates).
    - img1, img2: Images where points will be visualized (grayscale or color).
    - P1, P2: 3x4 camera projection matrices for the two views.

    Returns:
    - img1_points, img2_points: Images with 3D points visualized.
    """
    # Project 3D points to the two views
    points1 = (P1 @ np.column_stack((points_3d, np.ones(len(points_3d)))).T).T
    points2 = (P2 @ np.column_stack((points_3d, np.ones(len(points_3d)))).T).T
    points1 = points1[:, :2] / points1[:, 2].reshape(-1, 1)
    points2 = points2[:, :2] / points2[:, 2].reshape(-1, 1)
    # points1 = cv2.perspectiveTransform(points_3d.reshape(-1, 1, 3), P1).reshape(-1, 2)
    # points2 = cv2.perspectiveTransform(points_3d.reshape(-1, 1, 3), P2).reshape(-1, 2)

    # Draw points on the images
    img1_points = img1.copy()
    img2_points = img2.copy()

    for point1, point2 in zip(points1, points2):
        cv2.circle(img1_points, tuple(map(int, point1)), 5, (0, 255, 0), -1)
        cv2.circle(img2_points, tuple(map(int, point2)), 5, (0, 255, 0), -1)

    return img1_points, img2_points


def plot_3d_points(points_3d, filename):
    # Plot 3D points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        points_3d[:, 0],
        points_3d[:, 1],
        points_3d[:, 2],
    )
    ax.set_title("3D Points")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")  # type: ignore
    # set camera position
    ax.view_init(elev=-95, azim=-90)  # type: ignore
    # plt.show()
    plt.savefig(filename)
