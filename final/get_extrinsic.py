import glob
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
from get_label import parse_labels, data_path, data_dirname, dir_path
import camera_calibration_show_extrinsics as show


crop_pixels = 150
labels = parse_labels()

def crop_chessboard():
    zoomed_images = {}
    for filename, label in labels.items():
        if label != "No coordinates clicked":
            x, y = map(int, label.split(','))
            image_path = os.path.join(data_path, filename)
            image = cv2.imread(image_path)
            cropped = image[y-crop_pixels:y+crop_pixels, x-crop_pixels:x+crop_pixels]
            zoomed_images[filename] = cropped
            cv2.imshow('Cropped Image', cropped)
            cv2.waitKey(10)
            
    return zoomed_images

def load_images(images):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # (8,6) is for the given testing images.
    # If you use the another data (e.g. pictures you take by your smartphone),
    # you need to set the corresponding numbers.
    # corner_x = 7
    # corner_y = 7
    corner_x = 10
    corner_y = 7
    objp = np.zeros((corner_x * corner_y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = {}  # 2d points in image plane.

    # Step through the list and search for chessboard corners
    print("Start finding chessboard corners...")
    for filename, img in images.items():
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # plt.imshow(gray)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (corner_x, corner_y), None)
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints[filename] = corners

            # Draw and display the corners
            # cv2.drawChessboardCorners(img, (corner_x, corner_y), corners, ret)

        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        
    return imgpoints, objpoints

def resize_imgpoints(imgpoints):
    new_image_points = {}
    for filename, corners in imgpoints.items():
        center = labels[filename].split(',')
        center = (int(center[0]), int(center[1]))
        new_corners = corners
        offset = (center[0]-crop_pixels, center[1]-crop_pixels)
        for i in range(len(corners)):
            new_corners[i][0][0] += offset[0]
            new_corners[i][0][1] += offset[1]
        new_image_points[filename] = new_corners
    
    return imgpoints

def test_imgpoints(imgpoints):
    for filename, corners in imgpoints.items():
        image_path = os.path.join(data_path, filename)
        image = cv2.imread(image_path)
        cc = [0, 9, 60, 69]
        for c in cc:
            cv2.circle(image, (int(corners[c][0][0]), int(corners[c][0][1])), 10, (0, 0, 255), -1)
        # print(corners)
        image = cv2.resize(image, (1280, 720))
        cv2.imshow('img', image)
        cv2.waitKey(100)

    cv2.destroyAllWindows()
    
def compute_extrinsic(imgpoints, objpoints):
    # You need to comment these functions and write your calibration function from scratch.
    # Notice that rvecs is rotation vector, not the rotation matrix, and tvecs is translation vector.
    # In practice, you'll derive extrinsics matrixes directly. The shape must be [pts_num,3,4], and use them to plot.
    img_size = next(iter(imgpoints.values()))[0].shape[::-1]
    imgpoints = list(imgpoints.values())
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )
    Vr = np.array(rvecs)
    Tr = np.array(tvecs)
    extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1, 6)
    return mtx, extrinsics

def plot_extrinsics(mtx, extrinsics, name = "extrinsics.png"):
    name = os.path.join(data_path, name)
    # plot setting
    # You can modify it for better visualization
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # display
    # True -> fix board, moving cameras
    # False -> fix camera, moving boards
    min_values, max_values = show.draw_camera_boards(
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
    X_min = min_values[0]
    X_max = max_values[0]
    Y_min = min_values[1]
    Y_max = max_values[1]
    Z_min = min_values[2]
    Z_max = max_values[2]

    max_range = np.array([X_max - X_min, Y_max - Y_min, Z_max - Z_min]).max() / 2.0

    mid_x = (X_max + X_min) * 0.5
    mid_y = (Y_max + Y_min) * 0.5
    mid_z = (Z_max + Z_min) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, 0)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("-y")
    ax.set_title("Extrinsic Parameters Visualization")
    plt.show()
    # plt.savefig(name)

def write_extrinsic(filenames, extrinsics):
    extrinsics_file = os.path.join(data_path, 'extrinsics.txt')
    with open(extrinsics_file, 'w') as f:
        for filename, extrinsic in zip(filenames, extrinsics):
            f.write(f"{filename} ")
            for i in range(6):
                f.write(f"{extrinsic[i]} ")
            f.write('\n')

def write_intrinsics(mtx):
    intrinsics_file = os.path.join(data_path, 'intrinsics.txt')
    with open(intrinsics_file, 'w') as f:
        for i in range(3):
            for j in range(3):
                f.write(f"{mtx[i][j]} ")
            f.write('\n')

def write_imgpoints(imgpoints):
    imgpoints_file = os.path.join(data_path, 'imgpoints.txt')
    with open(imgpoints_file, 'w') as f:
        for filename, corners in imgpoints.items():
            f.write(f"{filename} ")
            f.write(f"{corners[0][0][0]} {corners[0][0][1]} ")
            f.write(f"{corners[9][0][0]} {corners[9][0][1]} ")
            f.write(f"{corners[60][0][0]} {corners[60][0][1]} ")
            f.write(f"{corners[69][0][0]} {corners[69][0][1]} ")
            f.write('\n')

def main():
    zoomed_images = crop_chessboard()

    imgpoints, objpoints = load_images(zoomed_images)
    imgpoints = resize_imgpoints(imgpoints)

    print(f'sucessful corners detected: {len(imgpoints)}')
    test_imgpoints(imgpoints)
    write_imgpoints(imgpoints)
    
    mtx, extrinsics = compute_extrinsic(imgpoints, objpoints)

    plot_extrinsics(mtx, extrinsics)

    write_extrinsic(imgpoints.keys(), extrinsics)
    write_intrinsics(mtx)

if __name__ == '__main__':
    main()