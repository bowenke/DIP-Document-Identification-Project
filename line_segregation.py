import cv2
import numpy as np
from numpy import random
from skimage import io
from skimage import img_as_float
from skimage import color
from itertools import product
from matplotlib import pyplot as plt


def load_img_from(path):
    # Will deal with stuff like cropping
    return color.rgb2gray(img_as_float(io.imread(path)))


def text_segmentation(image):
    x_grad_sq = np.sqrt((np.gradient(image, axis=0) ** 2))
    y_grad_sq = np.sqrt((np.gradient(image, axis=1) ** 2))
    intensity = x_grad_sq + y_grad_sq
    height, width = intensity.shape
    mean_intensity = np.sum(intensity, axis=0)
    print()

    # Compute for doldrums
    #doldrums = []
    #for mean in mean_intensity:





    visualize(intensity)


def visualize(image):
    plt.imshow(image, cmap='gray')
    plt.show()


def ransac_text_segmentation(image):
    y_grad = np.abs(np.gradient(image, axis=1))
    height, width = y_grad.shape
    random.seed(0)
    random.rand()


    #io.imshow(y_grad)
    #io.show()

    pass


if __name__ == '__main__':
    path = './c_w.png'
    image = load_img_from(path)
    text_segmentation(image)


