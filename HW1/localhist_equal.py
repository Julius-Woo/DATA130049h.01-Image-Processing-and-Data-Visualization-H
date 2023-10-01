from PIL import Image
from localhist_compute import local_histogram
import numpy as np

def local_histogram_equalization(img_path, L=256):

    '''
    Local histogram equalization based on 3x3 neighborhood.
    
    Parameters:
        - img_path: path to input image.
        - L: Number of intensity levels.
    Returns:
        - s_all: A 2D array consists of equalized intensity values in each position (i,j).
    '''
    
    # open image
    img = Image.open(img_path)
    
    # convert to grayscale if not already
    if img.mode != 'L':
        img = img.convert('L')

    # Get image size and pixels
    width, height = img.size
    pixels = img.load()

    # Calculate local histograms
    local_hist_array = local_histogram(img_path)
    
    # Create a 2D array to store the equalization result
    s_all = [[0 for _ in range(width)] for _ in range(height)]

    for i in range(height):
        for j in range(width):
            # Get the local histogram
            centerintensity = pixels[j, i]
            local_hist = local_hist_array[i][j]
            # Calculate the equalized value for the current pixel based on the local histogram
            s_all[i][j] = local_euqal_calc(local_hist, centerintensity, L)

    return s_all

def local_euqal_calc(local_hist, intensity, L=256):
    '''
    Modified histogram equalization based on given histogram dictionary with count.
    
    Parameters:
        - local_hist: Local histogram dictionary.
        - intensity: Intensity of the neighborhood center.
        - L: Number of intensity levels, default to 256.
    Returns:
        - s: equalized intensity of the neighborhood center.
    '''
    
    # Compute cumulative distribution function
    local_hist = dict(sorted(local_hist.items()))
    cdf = np.cumsum(list(local_hist.values()))
    
    # Calculate the equalized intensity s
    k = 0
    for i in local_hist.keys():
        if i == intensity:
            s = round((256-1)*cdf[k]/cdf[-1])
        k += 1
    return s

# test the function
# open image
img_path = '.\HW1\square_noise.tif'
img = Image.open(img_path)

# convert to grayscale if not already
if img.mode != 'L':
    img = img.convert('L')
    
# apply the function
pixels = img.load()
s_all = local_histogram_equalization(img_path)
for i in range(img.height):
    for j in range(img.width):
        pixels[j, i] = s_all[i][j]

# save the modified image
img_out_path = ".\HW1\square_local_equal.tif"
img.save(img_out_path)
print("Image saved to: {}".format(img_out_path))