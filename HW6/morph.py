from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def img_read(img_path):
    img = Image.open(img_path).convert('L')
    img = np.array(img)
    return img


def save_img(img, save_path):
    img = Image.fromarray(img)
    img.save(save_path)


def show_img(img):
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.show()


def dilation(image, kernel):
    k_h, k_w = kernel.shape
    i_h, i_w = image.shape

    # calculate padding size
    pad_height = k_h // 2
    pad_width = k_w // 2

    # padding image
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    dilated_image = np.zeros_like(image)
    for i in range(i_h):
        for j in range(i_w):
            # the neighborhood of the current pixel
            neighborhood = padded_image[i:i+k_h, j:j+k_w]
            # apply dilation operation
            dilated_image[i, j] = np.max(neighborhood * kernel)

    return dilated_image

def erosion(image, kernel):
    k_h, k_w = kernel.shape
    i_h, i_w = image.shape

    # calculate padding size
    pad_height = k_h // 2
    pad_width = k_w // 2

    # padding image
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    eroded_image = np.zeros_like(image)
    for i in range(i_h):
        for j in range(i_w):
            # the neighborhood of the current pixel
            neighborhood = padded_image[i:i+k_h, j:j+k_w]
            # apply erosion operation
            eroded_image[i, j] = np.min(neighborhood[kernel == 1])

    return eroded_image

def opening(image, kernel1, kernel2):
    return dilation(erosion(image, kernel1), kernel2)

def closing(image, kernel1, kernel2):
    return erosion(dilation(image, kernel1), kernel2)


# # test
# k = 5
# kernel = np.ones((k,k))

# binary_image = img_read('./zmic_fdu_noise.bmp')

# new_image = erosion(binary_image, kernel)
# show_img(new_image)
# new_image = dilation(binary_image, kernel)
# show_img(new_image)
# clear_img = opening(binary_image, np.ones((5,5)), np.ones((9,9)))
# show_img(clear_img)
# clear_img1 = closing(binary_image, np.ones((5, 5)), np.ones((9, 9)))
# show_img(clear_img1)