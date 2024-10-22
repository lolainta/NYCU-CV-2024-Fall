import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

from utils import get_images


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
    fig, axs = plt.subplots(
        2,
        len(pyramid),
        figsize=(4 * len(pyramid), 10),
    )
    # disable padding between subplots
    plt.subplots_adjust(hspace=0)

    # Show each level of the pyramid
    for i, layer in enumerate(pyramid):
        axs[0, i].imshow(layer, cmap="gray")
        layer_height, layer_width = layer.shape
        axs[0, i].set_title(f"Level {i}")
        axs[0, i].axis("off")

    for i, layer in enumerate(fft_amplitude):
        axs[1, i].imshow(layer)
        layer_height, layer_width = layer.shape
        axs[1, i].set_title(f"FFT Amplitude {i}")
        axs[1, i].text(
            0.5,
            -0.2,
            f"{layer_width}x{layer_height}",
            size=20,
            ha="center",
            transform=axs[1, i].transAxes,
        )
        axs[1, i].axis("off")

    plt.savefig(fname)


def process_image(img_name, img, level=5):
    gaussian_pyr = gaussian_pyramid(img, level)
    fft_amplitude = fft_pyramid(gaussian_pyr)
    save_pyramid(gaussian_pyr, fft_amplitude, f"results/pyramid/{img_name}.png")


def main():
    folder_path = "data/task1and2_hybrid_pyramid"
    images = get_images(folder_path)
    os.makedirs("results/pyramid", exist_ok=True)
    for img_name, img in tqdm(images):
        process_image(img_name, img)


if __name__ == "__main__":
    main()
