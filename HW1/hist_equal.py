from PIL import Image
import numpy as np

def histogram_equalization(img_path, L=256):
    '''
    global histogram equalization.
    
    Parameters:
        - img_path: path to input image.
        - L: Number of intensity levels.
    Returns:
        - s_dict: dictionary of equalized intensity values. The keys are original intensities, and the values are equalized intensities.
    '''

    r_dict = intensity_count(img_path,L)
    # calculate the cumulative distribution function
    cdf = np.cumsum(list(r_dict.values()))
    # calculate the equalized intensity s, in a dictionary
    s_dict = {i: round((L-1)*cdf[i]/cdf[-1]) for i in range(L)}
    
    return s_dict

def intensity_count(img_path, L=256):
    '''
    count the number of pixels for each intensity level.
    
    Parameters:
        - img_path: path to input image.
        - L: Number of intensity levels, default is 256.
    Returns:
        - count_dict: dictionary with intensity as key and count as value.
    '''
    # open image
    img = Image.open(img_path)
    
    # convert to grayscale if not already
    if img.mode != 'L':
        img = img.convert('L')
    
    # get pixel values
    pixels = list(img.getdata())

    # initialize dictionary
    count_dict = {i: 0 for i in range(L)}

    # count the number of pixels for each intensity
    for pixel in pixels:
        if pixel in count_dict:
            count_dict[pixel] += 1

    return count_dict


# test the function
# open image
img_path = '.\HW1\moon.tif'
img = Image.open(img_path)

# convert to grayscale if not already
if img.mode != 'L':
    img = img.convert('L')
    
# apply the function
pixels = img.load()
s = histogram_equalization(img_path)
for i in range(img.width):
    for j in range(img.height):
        intensity = pixels[i, j]
        pixels[i, j] = s[intensity]

# save the modified image
img_out_path = ".\HW1\moon_hist_equal.tif"
img.save(img_out_path)
print("Image saved to: {}".format(img_out_path))