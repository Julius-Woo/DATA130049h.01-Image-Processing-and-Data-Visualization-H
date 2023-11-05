from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def read_img(img_path):
    '''
    Read the image from the given path and convert it to grayscale.
    '''
    return np.array(Image.open(img_path).convert('L'))

def show_image(img):
    '''
    Show the image using matplotlib with axes.
    '''
    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap='gray')
    plt.show()

def img_modify(img, modified=0):
    '''
    Process the image for display based on the modification type.
    '''
    if modified==1:
        img = np.log(1+img)
        img = 255 * (img - np.min(img)) / (np.max(img) - np.min(img))
    elif modified==2:
        img = np.clip(img, 0, 255)
    elif modified==3:
        img = 255 * (img - np.min(img)) / (np.max(img) - np.min(img))
    return img.astype(np.uint8)

def show_spectrum(img):
    '''
    Calculate and show the frequency spectrum of the image.
    '''
    f = np.fft.fft2(img)
    f = np.fft.fftshift(f)
    spectrum = np.abs(f)
    spectrum = img_modify(spectrum, modified=1)

    plt.figure(figsize=(6,6))
    plt.imshow(spectrum, cmap='gray', norm=plt.Normalize())
    plt.show()
    
def show_spectrum2(f):
    '''
    Show the frequency spectrum from the given DFT.
    '''
    spectrum = np.abs(f)
    spectrum = img_modify(spectrum, modified=1)

    plt.figure(figsize=(6,6))
    plt.imshow(spectrum, cmap='gray', norm=plt.Normalize())
    plt.show()
    
def ghpf_shift(img, d0, u0, v0):
    '''
    Gaussian high pass filter (GHPF) with center shifted to (u0, v0).
    
    Parameters:
        - img: the input image, a 2D numpy array
        - d0: the cutoff frequency
        - u0, v0: the center coordinates of the highpass filter
        
    Returns:
        - filter_transfun: the filter transfer function of GHPF, with size m*n
    '''
    m, n = img.shape
    filter_transfun = np.zeros((m, n))
    for u in range(m):
        for v in range(n):
            d2 = (u-u0)**2 + (v-v0)**2
            filter_transfun[u, v] = 1 - np.exp(-d2/(2*d0**2))
    return filter_transfun

def notch_reject(img, coord, d0):
    '''
    Notch reject filter.
    
    Parameters:
        - img: the input image, a 2D numpy array
        - coord: the center coordinates of each highpass filter, k*2 array, k is the number of filters
        - d0: the cutoff frequency of the highpass filter
        
    Returns:
        - filter_transfun: the filter transfer function of notch reject filter, with size m*n
    '''
    m, n = img.shape
    k = coord.shape[0]
    nr = np.ones((m,n))
    for i in range(k):
        u, v = coord[i]
        nr *= ghpf_shift(img, d0, u, v) * ghpf_shift(img, d0, m-u, v) * ghpf_shift(img, d0, u, n-v) * ghpf_shift(img, d0, m-u, n-v)
    return nr

img_path='./hw4.png'
img = read_img(img_path)
# show_spectrum(img)

# # create the filter transfer function
xs = [16, 100, 180, 216, 300]
coor = np.array([[x, y] for x in xs for y in xs])
nr = notch_reject(img, coor, 21)

# apply the filter transfer function to the DFT of the image
f = np.fft.fft2(img)
f = np.fft.fftshift(f)
g = f * nr
show_spectrum2(g)

# do the inverse DFT and shift back
img_filtered = np.fft.ifftshift(g)
img_filtered = np.fft.ifft2(img_filtered)
img_filtered = np.real(img_filtered)

img_out = img_modify(img_filtered, modified=3)
show_image(img_out)

# Image.fromarray(img_out).save('brainct.png')