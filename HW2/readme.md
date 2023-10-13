# Codes involved in Homework 2

## Abstract
### Question2: 
`thres_otsu.py`

imports `global_hist.py` and `localhistW_compute.py`

Input image: `spot_shaded_text_image.tif`

Output image: `global_otsu.jpg` `local_otsu5.jpg` `local_otsu35.jpg` `local_otsu65.jpg` `thresh0.jpg`

### Question3: 
`linear_interpolation.py`

Input image: `chronometer 1250 dpi.tif`

Output image: `shrinked_image.jpg` and `shrinked_zoomed_image.jpg`


## localhistW_compute.py
The function in thid code calculates local histogram of each pixel with a W*W neighborhood efficiently using sliding window in a Z-shaped pattern.

### Parameters:
- img_path: path to input image.
- W: window size, an odd number. Default is 3.
### Returns:
- local_hist_array: 2D array of dictionaries where each dictionary is a histogram of the W*W neighborhood of a pixel.
    - local_hist_array[i][j] is a dictionary of the histogram of the W*W neighborhood of the pixel at (j, i).
    - The keys are the pixel values in the neighborhood and their values are the counts of those pixels. Exclude the zero-count pixels.

## global_hist.py
The `hist` function in this code calculates the global histogram of an image.

### Parameters:
- img_path: a grayscale image.
### Returns:
- hist: a histogram (a dictionary), where the key is the pixel value and the value is the frequency of that pixel value.


## thres_otsu.py
The code calculates the threshold value using Otsu's method. There are three functions in this code.
- `compute_otsu_threshold` function computes Otsu's threshold for a given input histogram.
- `adaptive_otsu` function implements adaptive Otsu's method and returns the thresholded image.
- `global_otsu` function implements global Otsu's method and also returns the thresholded image.


## linear_interpolation.py
The `resize_image` function in this code resizes an image using linear interpolation with the given scale factor.

### Parameters:
- img_path: path to the image
- scale: scale factor
### Returns:
- resized image