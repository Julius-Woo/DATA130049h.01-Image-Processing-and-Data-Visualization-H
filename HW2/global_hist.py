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

def histdraw(hist):
    """
    Draw the normalized histogram of an image.
    
    Input:
        - hist: a histogram (a dictionary), where the key is the pixel value and the value is the frequency of that pixel value.
    """
    import matplotlib.pyplot as plt
    N = sum(hist.values())
    for key in hist.keys():
        hist[key] /= N
    plt.bar(hist.keys(), hist.values(), color='g')
    plt.title('Normalized Global Histogram of Image')
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.show()