from PIL import Image

def piecewise_linear_transformation(img_path, r1, s1, r2, s2, L=256):
    
    '''
    Apply piecewise linear transformation for contrast stretching.
    
    Parameters:
        - img_path: path to input image.
        - r: input intensities.
        - s: output intensities.
        - r1, s1: First location point.
        - r2, s2: Second location point.
            Generally, r1<=r2 and s1<=s2.
        - L: Number of intensity levels, default is 256.
    Returns:
        - None. Saves the image and prints the path.
    '''

    # open image
    img = Image.open(img_path)
    
    # convert to grayscale if not already
    if img.mode != 'L':
        img = img.convert('L')

    # define piecewise linear function
    def transform_function(r,r1,r2,L):
        if r1 == 0:
            r1 = 1
        if r2 == L-1:
            r2 = L-2 # to avoid division by zero
        if r1 == r2:
            if 0 <= r < r1:
                s = round((s1 / r1) * r)
            else:
                s = round(((L-1 - s2) / (L-1 - r2)) * (r - r2) + s2)
        else:
            if 0 <= r < r1:
                s = round((s1 / r1) * r)
            elif r1 <= r <= r2:
                s = round(((s2 - s1) / (r2 - r1)) * (r - r1) + s1)
            else:
                s = round(((L-1 - s2) / (L-1 - r2)) * (r - r2) + s2)
        
        if s > L-1: # clamp to L-1
            return L-1
        else:
            return s
    
    # apply the function to every pixel value
    pixels = img.load()
    for i in range(img.width):
        for j in range(img.height):
            intensity = pixels[i, j]
            pixels[i, j] = transform_function(intensity,r1,r2,L)

    # save the modified image
    img_out_path = "piecewise_transed.tif"
    img.save(img_out_path)

    print(f"Image saved at {img_out_path}")
    
# Test the function
input_image_path = '.\HW1\pollen.tif'
img1 = Image.open(input_image_path)
pixels = img1.load()
maxr = -99
minr = 100
for i in range(img1.width):
    for j in range(img1.height):
        intensity = pixels[i, j]
        if intensity > maxr:
            maxr = intensity
        if intensity < minr:
            minr = intensity
print(minr,maxr)

piecewise_linear_transformation(input_image_path, minr, 0, maxr, 255)