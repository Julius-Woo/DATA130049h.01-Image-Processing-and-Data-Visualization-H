from PIL import Image
import numpy as np
import imgconvolution as ic

def sharpen_laplace(img):
    '''
    Sharpens an image using the Laplace operator.
    
    Parameters:
        - img: the image to be sharpened, a 2D numpy array
        
    Returns:
        - img_sharpened: the sharpened image
    '''
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    laplace_img = ic.convol(img, kernel)
    img_sharpened = img - laplace_img
    img_sharpened = np.clip(img_sharpened, 0, 255)
    return img_sharpened


def sharpen_masking(img, factor):
    '''
    Sharpens an image using high-boost filtering.

    Parameters:
        - img: the image to be sharpened, a 2D numpy array
        - factor: the boost factor. A value of 1 means no boost.

    Returns:
        - img_sharpened: the sharpened image
    '''
    import spatsmoothen
    
    blurred_img = spatsmoothen.smooth_gauss(img, 5)
    mask = img - blurred_img
    img_sharpened = img + factor * mask
    img_sharpened = np.clip(img_sharpened, 0, 255)
    return img_sharpened

# # test1
# img_path = './blurry_moon.tif'
# img = np.array(Image.open(img_path).convert('L'))

# img_sharpened1 = sharpen_laplace(img)

# new_img1 = Image.fromarray(img_sharpened1.astype('uint8'))

# new_img1.save('moon_laplace_sharpen.png')

# # test2
# img_path = './HW3/dipxe_text.tif'
# img = np.array(Image.open(img_path).convert('L'))

# img_sharpened2 = sharpen_laplace(img)
# img_sharpened3 = sharpen_masking(img, 1)
# img_sharpened4 = sharpen_masking(img, 4.5)

# new_img2 = Image.fromarray(img_sharpened2.astype('uint8'))
# new_img3 = Image.fromarray(img_sharpened3.astype('uint8'))
# new_img4 = Image.fromarray(img_sharpened4.astype('uint8'))

# new_img2.save('dipxe_text_laplace_sharpen.png')
# new_img3.save('dipxe_text_masking_sharpen.png')
# new_img4.save('dipxe_text_highboost_sharpen.png')