# Codes involved in Homework 3

## Abstract
### Question1(1):
`smoothen.py`

imports `imgconvolution.py`

Input image: `test_pattern_blurring.tif`

Output image: `pattern_box_smooth.png` and `pattern_gauss_smooth.png`

Input image: `ckt_board.tif`

Output image: `ckt_gauss_smooth.png` and `ckt_median_smooth.png`

### Question2(2): 
`sharpen.py`

imports `imgconvolution.py`, `smoothen.py`

Input image: `blurry_moon.tif`

Output image: `moon_laplace_sharpen.png`

Input image: `dipxe_text.tif`

Output image: `dipxe_text_laplace_sharpen.png`, `dipxe_text_masking_sharpen.png` and `dipxe_text_highboost_sharpen.png`

## `convol.py`

The `convol.py` module contains functions for convolving a 2D image with a given kernel.
### `convol`

The `convol` function performs convolution of a 2D image with a given kernel. This operation is commonly used for image filtering and feature extraction.

#### Parameters:

* `img` (numpy array): The input image to be convolved, represented as a 2D numpy array.
* `kernel` (numpy array): The kernel used for convolution.

#### Returns:

* `convol_img` (numpy array): The convolved image with floating-point values and the same size as the input image. The values are clipped between 0 and 255 to ensure they are within the valid pixel value range.

### `convol_full`

The `convol_full` function also performs convolution of a 2D image with a given kernel, but it uses the full convolution method. This method preserves the spatial dimensions of the image but results in an output image with larger dimensions.

#### Parameters:

* `img` (numpy array): The input image to be convolved, represented as a 2D numpy array.
* `kernel` (numpy array): The kernel used for convolution.

#### Returns:

* `convol_img` (numpy array): The convolved image with floating-point values. This version of the function pads the input image with zeros to accommodate the larger output dimensions.

## `smoothen.py`
### `smooth_box`

The `smooth_box` function applies a smoothing operation to an input image using a box filter of a specified size. This operation helps reduce noise and enhance the overall appearance of the image.
#### Parameters: 
- `img` (numpy array): The input image to be smoothed, represented as a 2D numpy array. 
- `size` (integer): The size of the box filter, which should be an odd integer.
#### Returns: 
- `img_smoothed` (numpy array): The resulting smoothed image after applying the box filter.
### `smooth_gauss`

The `smooth_gauss` function performs image smoothing using a Gaussian filter with a specified standard deviation. Gaussian smoothing is effective for reducing noise while preserving image features.
#### Parameters: 
- `img` (numpy array): The input image to be smoothed, represented as a 2D numpy array. 
- `sigma` (float): The standard deviation of the Gaussian filter.
#### Returns: 
- `img_smoothed` (numpy array): The smoothed image obtained by applying the Gaussian filter.
### `smooth_medianorder`

The `smooth_medianorder` function utilizes a median filter for image smoothing. Median filtering is effective in reducing salt-and-pepper noise while preserving edges.
#### Parameters: 
- `img` (numpy array): The input image to be smoothed, represented as a 2D numpy array. 
- `size` (integer): The size of the median filter, which should be an odd integer.
#### Returns: 
- `img_smoothed` (numpy array): The smoothed image achieved by applying the median filter.

## `sharpen.py`

### `sharpen_laplace`

The `sharpen_laplace` function is designed to sharpen an image using the Laplace operator. This technique enhances edges and fine details in the image.

#### Parameters:

* `img` (numpy array): The input image to be sharpened, represented as a 2D numpy array.

#### Returns:

* `img_sharpened` (numpy array): The resulting sharpened image after applying the Laplace operator.

### `sharpen_masking`

The `sharpen_masking` function sharpens an image using high-boost filtering, which enhances fine details while preserving edges. It does this by applying a mask to the image.

#### Parameters:

* `img` (numpy array): The input image to be sharpened, represented as a 2D numpy array.
* `factor` (float): The boost factor, where a value of 1 means no boost.

#### Returns:

* `img_sharpened` (numpy array): The sharpened image obtained by applying the high-boost filtering.