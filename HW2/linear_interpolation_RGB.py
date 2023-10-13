from PIL import Image

def linear_interpolation(val1, val2, alpha):
    '''
    Linear interpolation between val1 and val2
    
    Input:
        - val1: value 1
        - val2: value 2
        - alpha: interpolation factor
        
    Output:
        - interpolated value
    '''
    return val1 * (1 - alpha) + val2 * alpha

def resize_image(img_path, scale=1):
    '''
    Resize the image with the given scale
    
    Input:
        - img_path: path to the image
        - scale: scale factor
        
    Output:
        - resized image
    '''
    #  Open the image
    img = Image.open(img_path)
    width, height = img.size
    
    #  Calculate the new image size
    new_width = int(width * scale)
    new_height = int(height * scale)

    new_img = Image.new('RGB', (new_width, new_height))
    original_pixels = img.load()
    new_pixels = new_img.load()

    for x in range(new_width):
        for y in range(new_height):
            #  Map the pixel from new image to original image
            #  gx, gy: new pixels' coordinates in original image
            gx = x / scale
            gy = y / scale

            #  Get the coordinates of the 4 pixels around the new pixel
            gx0 = int(gx)
            gy0 = int(gy)
            gx1 = min(gx0 + 1, width - 1)
            gy1 = min(gy0 + 1, height - 1)

            #  Calculate the alpha values for interpolation
            alpha_x = gx - gx0
            alpha_y = gy - gy0

            # Linear interpolation for each channel
            interpolated_values = []
            for channel in range(3):  # For R, G, B channels
                val_y0 = linear_interpolation(original_pixels[gx0, gy0][channel], original_pixels[gx1, gy0][channel], alpha_x)
                val_y1 = linear_interpolation(original_pixels[gx0, gy1][channel], original_pixels[gx1, gy1][channel], alpha_x)
                interpolated_values.append(int(linear_interpolation(val_y0, val_y1, alpha_y)))

            new_pixels[x, y] = tuple(map(lambda v: max(0, min(255, v)), interpolated_values))  # Clamp each channel value to [0, 255]

    return new_img

# test
img_path = 'lowreso.jpeg'
resized_img = resize_image(img_path, 10)
resized_img.save("resized_low.jpeg")