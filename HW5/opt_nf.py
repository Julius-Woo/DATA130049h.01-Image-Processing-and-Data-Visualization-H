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
        nr *= ghpf_shift(img, d0, u, v) * ghpf_shift(img, d0, m-u, n-v)
    return nr

def notch_pass(img, coord, d0):
    '''
    Notch pass filter.
    '''
    return 1-notch_reject(img, coord, d0)

def optimum_notch(img, notch, m1, n1):
    '''
    Optimum notch filter.
    
    Parameters:
        - img: the input image, a 2D numpy array
        - notch: the notch filter transfer function, the same size as img
        - m1, n1: the size of the neighborhood, two odd integers
        
    returns:
        - img_filtered: the filtered image, a 2D numpy array
    '''
    def meanvalue(img, x, y, m1, n1):
        '''
        Calculate the mean value of the neighborhood of each pixel.
        - x,y: the center coordinates of the neighborhood
        '''
        m, n = img.shape
        img_mean = 0
        i1 = max(0, x-m1//2)
        i2 = min(m, x+m1//2+1)
        j1 = max(0, y-n1//2)
        j2 = min(n, y+n1//2+1)
        img_mean = np.mean(img[i1:i2, j1:j2])
        return img_mean
    
    m, n = img.shape
    w = np.zeros((m, n))
    
    # DFT of the image
    fg = np.fft.fft2(img)
    fg = np.fft.fftshift(fg)
    
    # interference pattern in sptaial domain
    eta = fg * notch
    eta = np.fft.ifftshift(eta)  
    eta = np.fft.ifft2(eta)
    eta = np.real(eta)
    
    # weighting function
    for x in range(m):
        for y in range(n):
            eta_bar = meanvalue(eta, x, y, m1, n1)
            eta2_bar = meanvalue(eta**2, x, y, m1, n1)
            geta_bar = meanvalue(img*eta, x, y, m1, n1)
            g_bareta_bar = meanvalue(img, x, y, m1, n1)*eta_bar
            w[x, y] = (geta_bar - g_bareta_bar) / (eta2_bar - eta_bar**2)
    
    # estimated f
    img_filtered = img - w * eta
    
    return img_filtered

# test
img_path = './noisy_image.png'
img = read_img(img_path)
m, n = img.shape
show_spectrum(img)

# create the filter transfer function
ys = np.arange(100, 37, -9)
xs = np.arange(105, -14, -17)
coor = np.column_stack((xs, ys))  # coordinates of bursts
k = coor.shape[0]
nr = np.ones((m,n))
for i in range(k):
    u, v = coor[i]
    nr *= ghpf_shift(img, 4.5, u, v) * ghpf_shift(img, 4.5, m-u, n-v)
nr *= ghpf_shift(img, 2, 137.8, 85) * ghpf_shift(img, 2, m-137.8, n-85)
notch = 1 - nr

img_filtered = optimum_notch(img, notch, 7, 5)
img_out = img_modify(img_filtered, modified=2)
show_spectrum(img_out)
show_image(img_out)

# Image.fromarray(img_out).save('brain_cleaned.png')