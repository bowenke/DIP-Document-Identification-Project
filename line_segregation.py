import numpy as np

from cv2 import ximgproc

import skimage
from skimage import io
from skimage import color
from skimage import filters

from itertools import product
from matplotlib import pyplot as plt


def load_img_from(path):
    image = io.imread(fname=path, as_grey=True)
    return image

def line_segmentation(image, save=False, visualize_lines=False):
    height, width = image.shape
    inten_thres = 0.1
    text_thres = 0.1
    print(image.shape)

    # Compute intensity and dismiss lightly tinted pixels are noises
    x_grad_sq = np.square(np.gradient(image, axis=1))
    x_grad_sq /= np.max(x_grad_sq)
    y_grad_sq = np.square(np.gradient(image, axis=0))
    y_grad_sq /= np.max(y_grad_sq)
    intensity = np.sqrt(x_grad_sq+y_grad_sq)
    for i, j in product(range(height), range(width)):
        if intensity[i][j] < 0.2: intensity[i][j] = 0

    # Percentage of how in a row are there pixels have intensity greater than 0.3
    energy_by_row = [len(row[row>inten_thres])/float(width) for row in intensity]

    # Compute horizontal doldrums
    doldrums = []
    temp = []
    for index, energy in enumerate(energy_by_row):
        if energy <= 0:
            temp.append(index)
        else:
            if len(temp) > 0:
                doldrums.append(int(np.average(temp)))
                temp = []
    if len(temp) > 0:
        doldrums.append(np.average(temp))
    print(doldrums)

    #for line in doldrums: image[int(line)] = np.asarray([[0.0] * width])
    #visualize(image)
    #return
    # Compute margins (hard margins)
    all_margins = []
    for line in doldrums:
        margins = [line, line]
        while int(margins[0]-1) >= 0 or int(margins[1]+1) < height:
            if np.sum(intensity[int(margins[0]-1)]) <= 0: margins[0] -= 1
            else: break
            if np.sum(intensity[int(margins[1]+1)]) <= 0: margins[1] += 1
            else: break
        all_margins = all_margins + margins
    #print(doldrums)
    #print(all_margins)

    del all_margins[0]
    del all_margins[-1]
    #if save_lines:
    #    for pair in all_margins:
    if visualize_lines:
        display = np.copy(image)
        for line in all_margins: display[int(line)] = np.asarray([[0.0] * width])
        visualize(display)

    return all_margins


def visualize(image):
    plt.imshow(image, cmap='gray')
    plt.show()

if __name__ == '__main__':
    path = './from_ceyer.png'
    image = load_img_from(path)
    line_segmentation(image, visualize_lines=True)


