import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from numpy import linspace
import cv2 as cv


def inverse_homogeneoux_matrix(M):
    R = M[0:3, 0:3]
    T = M[0:3, 3]
    M_inv = np.identity(4)
    M_inv[0:3, 0:3] = R.T
    M_inv[0:3, 3] = -(R.T).dot(T)

    return M_inv


def transform_to_matplotlib_frame(cMo, X, inverse=False):
    M = np.identity(4)
    M[1, 1] = 0
    M[1, 2] = 1
    M[2, 1] = -1
    M[2, 2] = 0

    if inverse:
        return M.dot(inverse_homogeneoux_matrix(cMo).dot(X))
    else:
        return M.dot(cMo.dot(X))


def create_camera_model(
    camera_matrix, width, height, scale_focal, draw_frame_axis=False
):
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    focal = 2 / (fx + fy)
    f_scale = scale_focal * focal

    # draw image plane
    X_img_plane = np.ones((4, 5))
    X_img_plane[0:3, 0] = [-width, height, f_scale]
    X_img_plane[0:3, 1] = [width, height, f_scale]
    X_img_plane[0:3, 2] = [width, -height, f_scale]
    X_img_plane[0:3, 3] = [-width, -height, f_scale]
    X_img_plane[0:3, 4] = [-width, height, f_scale]

    # draw triangle above the image plane
    X_triangle = np.ones((4, 3))
    X_triangle[0:3, 0] = [-width, -height, f_scale]
    X_triangle[0:3, 1] = [0, -2 * height, f_scale]
    X_triangle[0:3, 2] = [width, -height, f_scale]

    # draw camera
    X_center1 = np.ones((4, 2))
    X_center1[0:3, 0] = [0, 0, 0]
    X_center1[0:3, 1] = [-width, height, f_scale]

    X_center2 = np.ones((4, 2))
    X_center2[0:3, 0] = [0, 0, 0]
    X_center2[0:3, 1] = [width, height, f_scale]

    X_center3 = np.ones((4, 2))
    X_center3[0:3, 0] = [0, 0, 0]
    X_center3[0:3, 1] = [width, -height, f_scale]

    X_center4 = np.ones((4, 2))
    X_center4[0:3, 0] = [0, 0, 0]
    X_center4[0:3, 1] = [-width, -height, f_scale]

    # draw camera frame axis
    X_frame1 = np.ones((4, 2))
    X_frame1[0:3, 0] = [0, 0, 0]
    X_frame1[0:3, 1] = [f_scale / 2, 0, 0]

    X_frame2 = np.ones((4, 2))
    X_frame2[0:3, 0] = [0, 0, 0]
    X_frame2[0:3, 1] = [0, f_scale / 2, 0]

    X_frame3 = np.ones((4, 2))
    X_frame3[0:3, 0] = [0, 0, 0]
    X_frame3[0:3, 1] = [0, 0, f_scale / 2]

    if draw_frame_axis:
        return [
            X_img_plane,
            X_triangle,
            X_center1,
            X_center2,
            X_center3,
            X_center4,
            X_frame1,
            X_frame2,
            X_frame3,
        ]
    else:
        return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4]


def create_board_model(
    extrinsics, board_width, board_height, square_size, draw_frame_axis=False
):
    width = board_width * square_size
    height = board_height * square_size

    # draw calibration board
    X_board = np.ones((4, 5))
    # X_board_cam = np.ones((extrinsics.shape[0],4,5))
    X_board[0:3, 0] = [0, 0, 0]
    X_board[0:3, 1] = [width, 0, 0]
    X_board[0:3, 2] = [width, height, 0]
    X_board[0:3, 3] = [0, height, 0]
    X_board[0:3, 4] = [0, 0, 0]

    # draw board frame axis
    X_frame1 = np.ones((4, 2))
    X_frame1[0:3, 0] = [0, 0, 0]
    X_frame1[0:3, 1] = [height / 2, 0, 0]

    X_frame2 = np.ones((4, 2))
    X_frame2[0:3, 0] = [0, 0, 0]
    X_frame2[0:3, 1] = [0, height / 2, 0]

    X_frame3 = np.ones((4, 2))
    X_frame3[0:3, 0] = [0, 0, 0]
    X_frame3[0:3, 1] = [0, 0, height / 2]

    if draw_frame_axis:
        return [X_board, X_frame1, X_frame2, X_frame3]
    else:
        return [X_board]


def draw_camera_boards(
    ax,
    camera_matrix,
    cam_width,
    cam_height,
    scale_focal,
    extrinsics,
    board_width,
    board_height,
    square_size,
    patternCentric,
):
    min_values = np.inf
    max_values = -np.inf

    if patternCentric:
        X_moving = create_camera_model(
            camera_matrix, cam_width, cam_height, scale_focal
        )
        X_static = create_board_model(
            extrinsics, board_width, board_height, square_size
        )
    else:
        X_static = create_camera_model(
            camera_matrix, cam_width, cam_height, scale_focal, True
        )
        X_moving = create_board_model(
            extrinsics, board_width, board_height, square_size
        )

    cm_subsection = linspace(0.0, 1.0, extrinsics.shape[0])
    colors = [cm.jet(x) for x in cm_subsection]  # type: ignore

    for i in range(len(X_static)):
        X = np.zeros(X_static[i].shape)
        for j in range(X_static[i].shape[1]):
            X[:, j] = transform_to_matplotlib_frame(np.eye(4), X_static[i][:, j])
        ax.plot3D(X[0, :], X[1, :], X[2, :], color="r")
        min_values = np.minimum(min_values, X[0:3, :].min(1))
        max_values = np.maximum(max_values, X[0:3, :].max(1))

    for idx in range(extrinsics.shape[0]):
        cMo = np.eye(4, 4)
        if extrinsics.shape[1] == 6:
            R, _ = cv.Rodrigues(extrinsics[idx, 0:3])
            T = extrinsics[idx, 3:6]
        elif extrinsics.shape[1] == 3 and extrinsics.shape[2] == 4:
            R = extrinsics[idx, 0:3, 0:3]
            T = extrinsics[idx, 0:3, 3]
        else:
            print("Error extrinsic matrix size!")

        cMo[0:3, 3] = T
        cMo[0:3, 0:3] = R
        for i in range(len(X_moving)):
            X = np.zeros(X_moving[i].shape)
            for j in range(X_moving[i].shape[1]):
                X[0:4, j] = transform_to_matplotlib_frame(
                    cMo, X_moving[i][0:4, j], patternCentric
                )
            ax.plot3D(X[0, :], X[1, :], X[2, :], color=colors[idx])
            min_values = np.minimum(min_values, X[0:3, :].min(1))
            max_values = np.maximum(max_values, X[0:3, :].max(1))

    return min_values, max_values


def plot_extrinsics(mtx, extrinsics):
    # plot setting
    # You can modify it for better visualization
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # display
    # True -> fix board, moving cameras
    # False -> fix camera, moving boards
    min_values, max_values = draw_camera_boards(
        ax=ax,
        # camera setting
        camera_matrix=mtx,
        cam_width=0.064 / 0.1 * 50,
        cam_height=0.032 / 0.1 * 50,
        scale_focal=1600 * 500,
        extrinsics=extrinsics,
        # chess board setting
        board_width=8,
        board_height=6,
        square_size=20,
        # display
        patternCentric=True,
    )
    X_min = min_values[0]  # type: ignore
    X_max = max_values[0]  # type: ignore
    Y_min = min_values[1]  # type: ignore
    Y_max = max_values[1]  # type: ignore
    Z_min = min_values[2]  # type: ignore
    Z_max = max_values[2]  # type: ignore

    max_range = np.array([X_max - X_min, Y_max - Y_min, Z_max - Z_min]).max() / 2.0

    mid_x = (X_max + X_min) * 0.5
    mid_y = (Y_max + Y_min) * 0.5
    mid_z = (Z_max + Z_min) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, 0)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)  # type: ignore

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("-y")  # type: ignore
    ax.set_title("Extrinsic Parameters Visualization")

    plt.savefig("extrinsics.png")
    # animation for rotating plot

    # for angle in range(0, 360):
    #     ax.view_init(30, angle)
    #     plt.draw()
    #     plt.pause(0.001)
