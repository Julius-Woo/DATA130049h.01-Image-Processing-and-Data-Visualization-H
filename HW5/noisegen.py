from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift

def img_read(img_path):
    img = Image.open(img_path).convert('L')
    img = np.array(img)
    return img

def save_img(img, save_path):
    img = Image.fromarray(img)
    img.save(save_path)
    
def show_img(img):
    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap='gray')
    plt.show()

def show_spectrum(img):
    '''
    Calculate and show the frequency spectrum of the image.
    '''
    f = fft2(img)
    f = fftshift(f)
    spectrum = np.abs(f)
    spectrum = img_modify(spectrum, modified=1)

    plt.figure(figsize=(6, 6))
    plt.imshow(spectrum, cmap='gray', norm=plt.Normalize())
    plt.show()

def show_spectrum2(f):
    '''
    Show the frequency spectrum from the given DFT.
    '''
    spectrum = np.abs(f)
    spectrum = img_modify(spectrum, modified=1)

    plt.figure(figsize=(6, 6))
    plt.imshow(spectrum, cmap='gray', norm=plt.Normalize())
    plt.show()

def img_modify(img, modified=0):
    '''
    Process the image for display based on the modification type.
    '''
    if modified == 1:
        img = np.log(1+img)
        img = 255 * (img - np.min(img)) / (np.max(img) - np.min(img))
    elif modified == 2:
        img = np.clip(img, 0, 255)
    elif modified == 3:
        img = 255 * (img - np.min(img)) / (np.max(img) - np.min(img))
    return img.astype(np.uint8)

def white_noise_gen(shape, amplitude):
    '''
    Generate white noise.
    
    Parameters:
        - shape: the size of the noise, a tuple of (height, width).
        - spectrum: the spectrum of the noise, a constant.
    Returns:
        - noise: the generated noise, a numpy array.
    '''
    f = np.full(shape, amplitude)  # generate constant spectrum
    u0, v0 = (shape[0]//2, shape[1]//2)  # get the center of the spectrum
    
    phase = np.exp(2j * np.pi * np.random.random(shape))  # generate random phase
    f = f * phase  # generate half of the noise spectrum
    for i in range(u0+1, shape[0]):
        for j in range(shape[1]):
            if 0 <= 2*u0-i <= 255 and 0 <= 2*v0-j <= 255:
                f[i][j] = np.conj(f[2*u0-i][2*v0-j])
    f[u0, v0] = amplitude  # set the center of the spectrum to real
    noise = np.real(ifft2(ifftshift(f)))  # generate white noise in the spatial domain
    
    return noise

def gaussian_noise_gen(shape, mean, std):
    '''
    Generate gaussian noise.
    
    Parameters:
        - shape: the size of the noise, a tuple of (height, width).
        - mean: the mean of the gaussian distribution.
        - std: the standard deviation of the gaussian distribution.
    Returns:
        - noise: the generated noise, a numpy array.
    '''
    noise = np.random.normal(mean, std, shape)  # generate gaussian random number
    return noise

def rayleigh_noise_gen(shape, a, b):
    '''
    Generate raleigh noise.
    
    Parameters:
        - shape: the size of the noise, a tuple of (height, width).
        - a, b: parameters of the rayleigh distribution.
    Returns:
        - noise: the generated noise, a numpy array.
    '''
    u = np.random.uniform(0, 1, shape)  # generate uniform random number
    noise = a + np.sqrt(-b * np.log(1 - u))  # inverse transform sampling
    return noise

def img_interfere(img, noise=None, s_p=0 , ps=0, pp=0):
    '''
    Interfere the image with the noise.
    
    Parameters:
        - img: the image to be interfered, a numpy array.
        - noise: the noise to interfere the image, a numpy array. For salt and pepper noise, the input noise is None.
        - s_p: a flag to indicate whether the noise is salt and pepper noise. 0 for no, 1 for yes.
        - ps: the probability of salt noise.
        - pp: the probability of pepper noise.
    Returns:
        - img_interfered: the interfered image, a numpy array.
    '''
    if s_p ==0:
        img_interfered = img + noise
        return img_interfered
    else:
        # salt and pepper noise
        img_interfered = img.copy()
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if np.random.random() < ps:
                    img_interfered[i][j] = 255
                elif np.random.random() < pp:
                    img_interfered[i][j] = 0
        return img_interfered

# # test
# # brain
# img = img_read('./brainimg.png')

# brain_noised_white = img_interfere(img, white_noise_gen(img.shape, 100))
# # save_img(img_modify(white_noise_gen(img.shape, 100), 3), 'white_noise1.png')
# # save_img(img_modify(brain_noised_white, 2), 'brain_noised_white.png')

# brain_noised_gaussian = img_interfere(
#     img, gaussian_noise_gen(img.shape, 100, 10))
# # save_img(img_modify(brain_noised_gaussian, 2), 'brain_noised_gaussian.png')

# brain_noised_rayleigh = img_interfere(
#     img, rayleigh_noise_gen(img.shape, 120, 1000))
# save_img(img_modify(brain_noised_rayleigh, 2), 'brain_noised_rayleigh.png')

# brain_noised_s_p = img_interfere(img, s_p=1, ps=0.15, pp=0.08)
# # save_img(img_modify(brain_noised_s_p, 2), 'brain_noised_s_p.png')

# fig, axs = plt.subplots(2, 2, figsize=(7, 5))

# axs[0,0].imshow(img_modify(brain_noised_white, 2), cmap='gray')
# axs[0,0].set_title('White Noise')
# axs[0,0].axis('off')  

# axs[0,1].imshow(img_modify(brain_noised_gaussian), cmap='gray')
# axs[0,1].set_title('Gaussian Noise')
# axs[0,1].axis('off')  

# axs[1,0].imshow(img_modify(brain_noised_rayleigh, 2), cmap='gray')
# axs[1,0].set_title('Rayleigh Noise')
# axs[1,0].axis('off')  

# axs[1,1].imshow(img_modify(brain_noised_s_p, 2), cmap='gray')
# axs[1,1].set_title('Salt and Pepper Noise')
# axs[1,1].axis('off')  

# plt.show()


# # heart
# img = img_read('./heartimg.png')

# heart_noised_white = img_interfere(img, white_noise_gen(img.shape, 200))
# # save_img(img_modify(white_noise_gen(img.shape, 200), 3), 'white_noise2.png')
# # save_img(img_modify(heart_noised_white, 2), 'heart_noised_white.png')

# heart_noised_gaussian = img_interfere(
#     img, gaussian_noise_gen(img.shape, 80, 20))
# # save_img(img_modify(heart_noised_gaussian, 2), 'heart_noised_gaussian.png')

# heart_noised_rayleigh = img_interfere(img, rayleigh_noise_gen(img.shape, 150, 200))
# # save_img(img_modify(heart_noised_rayleigh, 2), 'heart_noised_rayleigh.png')

# heart_noised_s_p = img_interfere(img, s_p=1, ps=0.13, pp=0.27)
# # save_img(img_modify(heart_noised_s_p, 2), 'heart_noised_s_p.png')


# fig, axs = plt.subplots(2, 2, figsize=(7, 5))

# axs[0, 0].imshow(img_modify(heart_noised_white, 2), cmap='gray')
# axs[0, 0].set_title('White Noise')
# axs[0, 0].axis('off')

# axs[0, 1].imshow(img_modify(heart_noised_gaussian, 2), cmap='gray')
# axs[0, 1].set_title('Gaussian Noise')
# axs[0, 1].axis('off')

# axs[1, 0].imshow(img_modify(heart_noised_rayleigh, 2), cmap='gray')
# axs[1, 0].set_title('Rayleigh Noise')
# axs[1, 0].axis('off')

# axs[1, 1].imshow(img_modify(heart_noised_s_p, 2), cmap='gray')
# axs[1, 1].set_title('Salt and Pepper Noise')
# axs[1, 1].axis('off')

# plt.show()