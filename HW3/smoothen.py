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
    kernel = np.ones((size, size)) / (size * size)
    img_smoothed = ic.convol(img, kernel)
    img_smoothed = np.clip(img_smoothed, 0, 255)

    return img_smoothed


def smooth_gauss(img, sigma):
    """
    Smooths an image using a Gaussian filter with standard deviation sigma.
    
    Parameters:
        - img: the image to be smoothed, a 2D numpy array
        - sigma: the standard deviation of the Gaussian filter
        
    Returns:
        - img_smoothed: the smoothed image
    """
    size = int(6 * sigma)
    if size % 2 == 0:  # Ensure the size is odd
        size += 1
    
    pad_size = size//2
    # generate Gaussian kernel.
    x = np.arange(-pad_size, pad_size + 1)
    ker = np.exp(-0.5 * (x / sigma) ** 2)
    ker = ker / np.sum(ker)
    ker = np.outer(ker, ker)
    img_smoothed = ic.convol(img, ker)
    img_smoothed = np.clip(img_smoothed, 0, 255)
    
    return img_smoothed

def smooth_medianorder(img, size):
    '''
    Smooths an image using a median filter.
    
    Parameters:
        - img: the image to be smoothed, a 2D numpy array
        - size: the size of the filter, an odd integer
        
    Returns:
        - img_smoothed: the smoothed image
    '''
    if size % 2 == 0:
        raise ValueError("size must be an odd integer")
    
    h, w = img.shape
    pad_size = size // 2
    img_padded = np.pad(img, pad_size, mode='edge')
    img_smoothed = np.zeros_like(img)
    
    for i in range(h):
        for j in range(w):
            img_smoothed[i, j] = np.median(img_padded[i:i+size, j:j+size])
            
    return img_smoothed

# # test1
# img_path = './test_pattern_blurring.tif'
# img = np.array(Image.open(img_path).convert('L'))

# img_smoothed1 = smooth_box(img, 15)
# img_smoothed2 = smooth_gauss(img, 7)

# new_img1 = Image.fromarray(img_smoothed1.astype('uint8'))
# new_img2 = Image.fromarray(img_smoothed2.astype('uint8'))

# new_img1.save('pattern_box_smooth.png')
# new_img2.save('pattern_gauss_smooth.png')

# # test2
# img_path = './ckt_board.tif'
# img = np.array(Image.open(img_path).convert('L'))

# img_smoothed3 = smooth_gauss(img, 3)
# img_smoothed4 = smooth_medianorder(img, 7)


# new_img3 = Image.fromarray(img_smoothed3.astype('uint8'))
# new_img4 = Image.fromarray(img_smoothed4.astype('uint8'))

# new_img3.save('ckt_gauss_smooth.png')
# new_img4.save('ckt_median_smooth.png')