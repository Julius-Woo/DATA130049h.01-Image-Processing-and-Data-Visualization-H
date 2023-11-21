from PIL import Image
import matplotlib.pyplot as plt


def hist(img_path):
    """
    Calculate the global histogram of an image.
    
    Input:
        - img_path: a grayscale image.
        
    Output:
        - hist: a histogram (a dictionary), where the key is the pixel value and the value is the frequency of that pixel value.
    """
    from PIL import Image
    img = Image.open(img_path)

    if img.mode != 'L':
        img = img.convert('L')

    width, height = img.size
    pixels = img.load()

    hist = {}
    for i in range(height):
        for j in range(width):
            pixel = pixels[j, i]
            if pixel not in hist:
                hist[pixel] = 1
            else:
                hist[pixel] += 1
    return hist

def compute_otsu_threshold(hist):
    """
    Compute Otsu's threshold for a given histogram.
    
    Input:
        - hist: a histogram (a dictionary), where the key is the pixel value and the value is the frequency of that pixel value.
        
    Output:
        - threshold: the optimal threshold value.
    """
    # If only one value in histogram
    if len(hist) == 1:
        return 0 if next(iter(hist)) < 128 else 255
    
    #  extract the sorted keys from the histogram
    keys = sorted(hist.keys())
    
    #  compute the total number of pixels in the neighborhood
    total = sum(hist.values())
    
    # the mean of the image
    mG = sum([i * hist[i] for i in hist])/ total
    
    # initialize the variables
    var_between = 0
    threshold = 0
    max_var = -float('inf')
    best_ks = []  # list to store all k values with max variance
    p1_cumulative = 0
    m_cumulative = 0

    for index in range(0, len(keys)-1):
        k = keys[index]
        post_k = keys[index + 1]
        
        # Update the cumulative sums for the next iteration
        p1_cumulative += hist[k] / total
        m_cumulative += k * hist[k] / total
        
        p1 = p1_cumulative  # the weight background
        m = m_cumulative  #  the cumulative mean background
        if p1 == 0 or p1 == 1:
            continue
        
        # Check if the current variance is greater than max_var
        var_between = (mG*p1-m)**2/(p1*(1-p1))  # the variance between classes
        if var_between > max_var:
            max_var = var_between
            best_ks = [i for i in range(k, post_k)]
        elif var_between == max_var:
            best_ks = best_ks + [i for i in range(k, post_k)]
        
    # Compute the average threshold from all best k values
    if len(best_ks) > 0:
        threshold = sum(best_ks) / len(best_ks)
    return threshold

def global_otsu(img_path):
    '''
    Global Otsu's thresholding algorithm.
    
    Input:
        - img_path: the path to the image.
        
    Output:
        - output: the binarized image.
    '''
    
    histogram = hist(img_path)
    threshold = compute_otsu_threshold(histogram)
    
    img = Image.open(img_path)
    if img.mode != 'L':
        img = img.convert('L')

    width, height = img.size
    pixels = img.load()

    output = Image.new('L', (width, height))
    output_pixels = output.load()
    
    for i in range(height):
        for j in range(width):
            output_pixels[j, i] = 255 if pixels[j, i] > threshold else 0

    return output

# # test
# img_path = './heart_noised.png'

# result = global_otsu(img_path)
# plt.figure(figsize=(6, 6))
# plt.imshow(result, cmap='gray')
# plt.show()
# # result.save('./otsu_bin.png')

# print('Done!')