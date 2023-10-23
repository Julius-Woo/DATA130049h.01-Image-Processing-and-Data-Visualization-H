import numpy as np

def convol(img, kernel):
    """
    Convolves a 2D image with a kernel.
    
    Parameters:
        - img: the image to be convolved, a 2D numpy array
        - kernel: the kernel to convolve the image with
        
    Returns:
        - the convolved image numpy array with float values and the same size as the input image
    """
    if img.ndim != 2:
        raise ValueError('The image must be a 2D numpy array.')
    h, w = img.shape
    
    # Ensure the kernel is 2D
    if kernel.ndim == 1:
        kernel = kernel.reshape((1, -1))

    k_h, k_w = kernel.shape
    convol_img = np.zeros((h, w))
    rot_k = np.rot90(np.rot90(kernel))  # rotate the kernel 180 degrees
    
    # pad the image with zeros
    pad_h = int((k_h - 1)/2)
    pad_w = int((k_w - 1)/2)
    pad_img = np.zeros((h + 2 * pad_h, w + 2 * pad_w))
    pad_img[pad_h:pad_h + h, pad_w:pad_w + w] = img
    
    # convolve the image
    for i in range(h):
        for j in range(w):
            value = np.sum(pad_img[i:i + k_h, j:j + k_w] * rot_k)
            if value > 255:
                value = 255
            elif value < 0:
                value = 0
            convol_img[i,j] = value
            
    return convol_img


def convol_full(img, kernel):
    """
    Convolves a 2D image with a kernel using the full convolution method.
    
    Parameters:
        - img: the image to be convolved, a 2D numpy array
        - kernel: the kernel to convolve the image with
        
    Returns:
        - the convolved image numpy array with float values and padded zeros
    """
    if img.ndim != 2:
        raise ValueError('The image must be a 2D numpy array.')
    h, w = img.shape
    
    # Ensure the kernel is 2D
    if kernel.ndim == 1:
        kernel = kernel.reshape((1, -1))

    k_h, k_w = kernel.shape
    convol_img = np.zeros((h+k_h-1, w+k_w-1))
    rot_k = np.rot90(np.rot90(kernel))  # rotate the kernel 180 degrees
    
    # pad the image with zeros
    pad_h = k_h - 1
    pad_w = k_w - 1
    pad_img = np.zeros((h + 2 * pad_h, w + 2 * pad_w))
    pad_img[pad_h:pad_h + h, pad_w:pad_w + w] = img
    
    # convolve the image
    for i in range(h+k_h-1):
        for j in range(w+k_w-1):
            value = np.sum(pad_img[i:i + k_h, j:j + k_w] * rot_k)
            if value > 255:
                value = 255
            elif value < 0:
                value = 0
            convol_img[i,j] = value
            
    return convol_img