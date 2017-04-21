import numpy as np

from cv2 import ximgproc

from numpy import random
import skimage
from skimage import io
from skimage import color
from skimage import filters

from itertools import product
from matplotlib import pyplot as plt


def load_img_from(path):
    image = io.imread(fname=path, as_grey=True)
    return image


def text_segmentation(image):
    height, width = image.shape
    print(image.shape)
    x_grad_sq = np.square(np.gradient(image, axis=1))
    x_grad_sq /= np.max(x_grad_sq)
    y_grad_sq = np.square(np.gradient(image, axis=0))
    y_grad_sq /= np.max(y_grad_sq)
    intensity = np.sqrt(x_grad_sq+y_grad_sq)
    #visualize(intensity)

    # How to deal with situations when the there are many i and j s in the sentence?
    energy_by_row = [len(row[row>0.2])/float(width) for row in intensity]

    # Compute doldrums
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

    for line in doldrums: image[int(line)] = np.asarray([[0.0] * width])
    visualize(image)
    '''
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
    #for line in all_margins: image[int(line)] = np.asarray([[0.0] * width])
    del all_margins[0]
    del all_margins[-1]
    print(all_margins)
    #while len(all_margins) > 0:
    #io.imsave(fname='test_{}_{}'.format(all_margins[0], all_margins[1]), arr=image[14:73])
    return all_margins
    '''
def image_padding(image_set):
    # Compute for maximum padding
    # Apply paddings
    pass



def visualize(image):
    plt.imshow(image, cmap='gray')
    plt.show()

if __name__ == '__main__':
    path = './from_ceyer.png'
    image = load_img_from(path)
    text_segmentation(image)


