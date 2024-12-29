import cv2
import numpy as np
import os
from tqdm import trange, tqdm
from icecream import ic
from multiprocessing import Pool
from jax import numpy as jnp


from camera_calibration_show_extrinsics import plot_extrinsics

np.set_printoptions(suppress=True, linewidth=200, threshold=10)
ic.configureOutput(includeContext=True)
# ic.disable()

CASE = "room"
CASE = "park"
CASE = "park-2"
CASE = "park-3"

DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
META_DIR = os.path.join(DATA_DIR, CASE)


OUTPUT_DIR = os.path.join("output", CASE)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def align_checkboard(images, checkboard_corners):

    PANO_WIDTH = 8000
    PANO_HEIGHT = 4000
    MIN_WIDTH = PANO_WIDTH // 2 - 100
    MAX_WIDTH = PANO_WIDTH // 2 + 100
    MIN_HEIGHT = PANO_HEIGHT // 2 - 100
    MAX_HEIGHT = PANO_HEIGHT // 2 + 100

    fixed_cords = [
        [MIN_WIDTH, MIN_HEIGHT],
        [MAX_WIDTH, MIN_HEIGHT],
        [MIN_WIDTH, MAX_HEIGHT],
        [MAX_WIDTH, MAX_HEIGHT],
    ]

    if CASE == "room":
        fixed_cords = fixed_cords[0], fixed_cords[2], fixed_cords[1], fixed_cords[3]
    elif CASE == "park":
        fixed_cords = fixed_cords[3], fixed_cords[2], fixed_cords[1], fixed_cords[0]
    elif CASE == "park-2":
        fixed_cords = fixed_cords[0], fixed_cords[2], fixed_cords[1], fixed_cords[3]
    elif CASE == "park-3":
        fixed_cords = fixed_cords[0], fixed_cords[2], fixed_cords[1], fixed_cords[3]
    else:
        raise ValueError("Invalid CASE")

    # add static args
    def align(img_name, image, corner, fixed_cords):
        # ic(img_name)
        M = cv2.getPerspectiveTransform(
            np.array(corner, dtype=np.float32),
            np.array(fixed_cords, dtype=np.float32),
        )

        # ic(M)
        ret = cv2.warpPerspective(image, M, (PANO_WIDTH, PANO_HEIGHT))
        # ic(ret.shape)
        cv2.imwrite(f"{OUTPUT_DIR}/{img_name}", ret)
        return ret

    for i, ((img_name, image), corner) in enumerate(zip(images, checkboard_corners)):
        ic(img_name)
        align(img_name, image, corner, fixed_cords)

    pano = jnp.zeros((PANO_HEIGHT, PANO_WIDTH, 3, len(images)), dtype=np.uint8)
    for i, (img_name, _) in tqdm([*enumerate(images)]):
        img = cv2.imread(f"{OUTPUT_DIR}/{img_name}")
        pano = pano.at[..., i].set(img)

    print("Merging images...")
    panorama = jnp.true_divide(pano.sum(axis=-1), (pano != 0).sum(axis=-1))
    panorama = np.array(panorama)
    cv2.imwrite(f"{OUTPUT_DIR}/panorama.jpg", panorama)

    print(f"Panorama saved to {OUTPUT_DIR}/panorama.jpg")


def main():
    IMGPOINTS_PATH = os.path.join(META_DIR, "imgpoints.txt")
    checkboard_pair = [
        (line.split()[0], np.array(line.split()[1:], dtype=np.float32))
        for line in open(IMGPOINTS_PATH).readlines()
    ]
    images = [
        (fname, cv2.imread(os.path.join(IMAGE_DIR, fname), cv2.IMREAD_COLOR))
        for fname, _ in checkboard_pair
    ]

    checkboard_corners = np.array(
        [np.array(pts).reshape(-1, 2) for _, pts in checkboard_pair]
    )
    # ic(checkboard_corners)
    print("Case:", CASE)
    align_checkboard(images, checkboard_corners)


if __name__ == "__main__":
    main()
    
