import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from utils import get_images, get_image_pairs

# FOLDER_PATH = "./data/task1and2_hybrid_pyramid/"
FOLDER_PATH = "./my_data/task1and2_hybrid_pyramid/"


def getGaussianKernel(kernel_size, sigma):
    X, Y = np.meshgrid(np.arange(kernel_size), np.arange(kernel_size))
    kernel = np.exp(
        -((X - kernel_size // 2) ** 2 + (Y - kernel_size // 2) ** 2) / (2 * sigma**2)
    )
    return kernel / kernel.sum()


def gaussian_smooth(img, kernel_size=3, sigma=np.sqrt(0.5)):
    kernel = getGaussianKernel(kernel_size, sigma)
    rows, cols = img.shape
    dis = kernel_size // 2
    ret = np.zeros((rows, cols))
    for i in range(dis, rows - dis):
        for j in range(dis, cols - dis):
            ret[i, j] = np.sum(
                img[i - dis : i + dis + 1, j - dis : j + dis + 1] * kernel
            )
    return ret


def subsample(img):
    return img[::2, ::2]


def gaussian_pyramid(img, levels):
    pyramid = [img]
    for _ in range(levels - 1):
        img = gaussian_smooth(img)
        img = subsample(img)
        pyramid.append(img)
    return pyramid


def fft_pyramid(layers):
    fft_list = []
    for layer in layers:
        fft = np.fft.fft2(layer)
        fft = np.fft.fftshift(fft)
        fft = np.log(np.abs(fft))
        fft_list.append(fft)
    return fft_list


# Function to display pyramids
def save_pyramid(pyramid, fft_amplitude, fname: str):
    plt.cla()
    m, n = pyramid[0].shape
    ratio = m / n
    fig, axs = plt.subplots(
        2,
        len(pyramid),
        figsize=(2 * len(pyramid), 5 * ratio),
        constrained_layout=True,
    )
    # disable padding between subplots
    # fig.frameon = False

    # Show each level of the pyramid
    for i, layer in enumerate(pyramid):
        axs[0, i].imshow(layer, cmap="gray")
        layer_height, layer_width = layer.shape
        axs[0, i].set_title(f"Level {i}")
        axs[0, i].axis("off")

    for i, layer in enumerate(fft_amplitude):
        axs[1, i].imshow(layer)
        layer_height, layer_width = layer.shape
        axs[1, i].text(
            0.5,
            -0.2,
            f"{layer_width}x{layer_height}",
            ha="center",
            transform=axs[1, i].transAxes,
        )
        axs[1, i].axis("off")

    plt.savefig(fname)


def process_image(img_name, img, level=5):
    gaussian_pyr = gaussian_pyramid(img, level)
    fft_amplitude = fft_pyramid(gaussian_pyr)
    return gaussian_pyr, fft_amplitude


def main():
    os.makedirs("./output/pyramid", exist_ok=True)
    images = get_images(FOLDER_PATH)
    for img_name, img in tqdm(images):
        tqdm.write(f"Processing {img_name}")
        gpyr, fpyr = process_image(img_name, img)
        save_pyramid(gpyr, fpyr, f"./output/pyramid/{img_name[:-4]}.png")


def expirements():
    import cv2

    os.makedirs("./output/pyramid", exist_ok=True)
    imgs = get_image_pairs("./output/pyramid")
    for num, (l_img, r_img) in imgs:
        if l_img.shape != r_img.shape:
            print(f"Resizing {r_img.shape} to match {l_img.shape}")
            r_img = cv2.resize(r_img, (l_img.shape[1], l_img.shape[0]))
        merged = np.hstack((l_img, r_img))
        cv2.imwrite(f"./output/pyramid/result_{num}.png", merged)


if __name__ == "__main__":
    main()
    expirements()
