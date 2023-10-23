from PIL import Image
import numpy as np
import imgconvolution as ic

def smooth_box(img, size):
    """
    Smooths an image using a box filter of size size * size.
    
    Parameters:
        - img: the image to be smoothed, a 2D numpy array
        - size: the size of the box filter, an odd integer
        
    Returns:
        - img_smoothed: the smoothed image
    """
    # kernel = np.ones((size, size)) / (size * size)
    # img_smoothed = ic.convol(img, kernel)
    
    h, w = img.shape
    pad_size = int((size - 1)/2)
    ker = np.ones((1, size))/size
    img_smoothed = ic.convol_full(img, ker)
    img_smoothed = ic.convol_full(img_smoothed, ker.T)[pad_size:pad_size + h, pad_size:pad_size + w]

    return img_smoothed


def smooth_gauss(img, sigma):
    """
    Smooths an image using a Gaussian filter with standard deviation sigma.
    
    Parameters:
        - img: the image to be smoothed
        - sigma: the standard deviation of the Gaussian filter
        
    Returns:
        - img_smoothed: the smoothed image
    """
    size = int(6 * sigma) + 1
    if size % 2 == 0:  # Ensure the size is odd
        size += 1
    
    h, w = img.shape
    pad_size = size//2
    # generate a 1D Gaussian kernel.
    x = np.arange(-pad_size, pad_size + 1)
    ker = np.exp(-0.5 * (x / sigma) ** 2)
    ker = ker / np.sum(ker)
    # ker = np.outer(ker, ker)
    # img_smoothed = ic.convol(img, ker)
    
    img_smoothed = ic.convol_full(img, ker)
    img_smoothed = ic.convol_full(img_smoothed, ker.T)
    img_smoothed = img_smoothed[pad_size:pad_size + h, pad_size:pad_size + w]
    
    return img_smoothed


# test
img_path = './HW3/test_pattern_blurring.tif'
# img_path = './HW3/ckt_board.tif'
img = np.array(Image.open(img_path).convert('L'))
# new_img = Image.fromarray(smooth_box(img, 3).astype('uint8'))
# img_smoothed = smooth_box(img, 7)

# convert the image to uint8
img_smoothed = smooth_gauss(img, 7)
img_smoothed = np.clip(img_smoothed, 0, 255)
new_img = Image.fromarray(img_smoothed.astype('uint8'))

new_img.show()