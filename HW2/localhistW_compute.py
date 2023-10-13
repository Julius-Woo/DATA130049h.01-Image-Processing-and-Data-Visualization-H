from PIL import Image

def local_histogram(img_path, W=3):
    '''
    Calculates local histogram of each pixel with a W*W neighborhood efficiently using sliding window in a Z-shaped pattern.

    Parameters:
        - img_path: path to input image.
        - W: window size, an odd number. Default is 3.
    Returns:
        - local_hist_array: 2D array of dictionaries where each dictionary is a histogram of the W*W neighborhood of a pixel.
            - local_hist_array[i][j] is a dictionary of the histogram of the W*W neighborhood of the pixel at (j, i).
                    The keys are the pixel values in the neighborhood and their values are the counts of those pixels. Exclude the zero-count pixels.
    '''
    img = Image.open(img_path)
    if img.mode != 'L':
        img = img.convert('L')

    width, height = img.size
    pixels = img.load()
    
    w = W // 2  # Half of the window size

    # Create a 2D array to store the local histograms
    local_hist_array = [[{} for _ in range(width)] for _ in range(height)]

    i, j = 0, 0
    while 0 <= i < height and 0 <= j < width:
        if i == 0 and j == 0:  # Initialize the first histogram manually
            histogram = {}
            for x in range(w+1):
                for y in range(w+1):
                    if i + x < height and j + y < width:
                        pixel_value = pixels[j + y, i + x]
                        histogram[pixel_value] = histogram.get(pixel_value, 0) + 1
            local_hist_array[i][j] = histogram

        # Move to the next pixel
        j = j + 1 if i % 2 == 0 else j - 1
        
        # Check if we need to move down or continue horizontally
        move_down = False
        if i % 2 == 0:
            if j == width:  # Reach the end of even row
                i += 1
                j -= 1
                move_down = True
        else:
            if j == -1:  # Reach the start of odd row
                i += 1
                j += 1
                move_down = True
                
        # Update the histogram based on the movement
        if move_down:
            if i == height:  # Reach the end of the image
                break
            else:  # Move down
                histogram = local_hist_array[i-1][j].copy()
                # Adjust for the row that's moved out of the window
                if i-w-1 >= 0:
                    for y in range(-w, w+1):
                        nj = j + y
                        if 0 <= nj < width:
                            pixel_value = pixels[nj, i-w-1]
                            histogram[pixel_value] -= 1
                            if histogram[pixel_value] == 0:
                                del histogram[pixel_value]

                # Adjust for the row that's moved into the window
                if i+w < height:
                    for y in range(-w, w+1):
                        nj = j + y
                        if 0 <= nj < width:
                            pixel_value = pixels[nj, i+w]
                            histogram[pixel_value] = histogram.get(pixel_value, 0) + 1

            
        else:  # Continue horizontally
            if i % 2 == 0:  # Even row
                prev_j = j - 1
                remove_col = j - w - 1
                add_col = j + w
            else:  # Odd row
                prev_j = j + 1
                remove_col = j + w + 1
                add_col = j - w
        
            histogram = local_hist_array[i][prev_j].copy()
            
            # Adjust for the column that's moved out of the window
            if 0 <= remove_col < width: # Check if we do not go out of bounds
                for x in range(-w, w+1):
                    ni = i + x
                    if 0 <= ni < height:
                        pixel_value = pixels[remove_col, ni]
                        histogram[pixel_value] -= 1
                        if histogram[pixel_value] == 0:
                            del histogram[pixel_value]

            # Adjust for the column that's moved into the window
            if 0 <= add_col < width:  # Check if we do not go out of bounds
                for x in range(-w, w+1):
                    ni = i + x
                    if 0 <= ni < height:
                        pixel_value = pixels[add_col, ni]
                        histogram[pixel_value] = histogram.get(pixel_value, 0) + 1

        local_hist_array[i][j] = histogram
    
    return local_hist_array