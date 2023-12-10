from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import loc_affine as af

def save_img(img, save_path):
    img = Image.fromarray(img)
    img.save(save_path)

def bilin(image, i, j):
    """
    Perform bilinear interpolation for a given point.

    :param image: Input color image as a NumPy array with shape (rows, cols, 3).
    :param i, j: The row, column coordinates of the point for interpolation.
    :return: Interpolated 3-channel value at point (i, j).
    """
    rows, cols, _ = image.shape
    # Calculate the integer parts of i and j
    up = int(np.floor(i))
    down = int(np.ceil(i))
    left = int(np.floor(j))
    right = int(np.ceil(j))
    # Return a zero vector for out-of-bound coordinates
    if up < 0 or left < 0 or down >= rows or right >= cols:
        return np.zeros(3)
    # Calculate the differences
    di = i - up
    dj = j - left
    # Perform bilinear interpolation for each channel
    new_value = np.zeros(3)
    for c in range(3):  # Loop over each color channel
        new_value[c] = (di * dj * image[up, left, c] +
                                 di * (1 - dj) * image[up, right, c] +
                                 (1 - di) * dj * image[down, left, c] +
                                 (1 - di) * (1 - dj) * image[down, right, c])
    return new_value

def back_trans(img, x_d, y_d, x_s, y_s, e):
    '''
    Backward transformation
    :param img: source image
    :param x_d, y_d: destination points' coordinates, each is an array.
    :param x_s, y_s: source points' coordinates, each is an array.
    :param e: exponent in weight calculation
    '''
    new_img = np.zeros_like(img)  # destination image
    h, w, _ = img.shape
    # get affine transformation matrix T
    T = af.get_T(x_d, y_d, x_s, y_s)
    for i in range(h):
        for j in range(w):
            # get new coordinates
            new_i, new_j = af.get_transcorr(i, j, x_d, y_d, T, e)
            # bilinear interpolation
            new_img[i, j] = bilin(img, new_i, new_j)
    
    return new_img


class ControlPointsSelector:
    def __init__(self, img1, img2):
        self.fig, self.axs = plt.subplots(1, 2)
        self.axs[0].imshow(img1)
        self.axs[1].imshow(img2)
        self.axs[0].set_title('Source Image')
        self.axs[1].set_title('Reference Image')

        self.x_s, self.y_s = [], []
        self.x_d, self.y_d = [], []
        self.selecting_source = True  # Start by selecting from the left image

        self.status_text = self.fig.text(0.05, 0.95, 'Select a point on the left image.',
                                         transform=self.fig.transFigure,
                                         ha="left", va="top", color="black")
        self.cid = self.fig.canvas.mpl_connect(
            'button_press_event', self.onclick)

    def onclick(self, event):
        ax_index = 0 if self.selecting_source else 1
        if event.inaxes == self.axs[ax_index]:
            x, y = round(event.xdata), round(event.ydata)

            if self.selecting_source:
                self.x_s.append(x)
                self.y_s.append(y)
                self.axs[0].plot(x, y, 'ro')  # Add red dot
            else:
                self.x_d.append(x)
                self.y_d.append(y)
                self.axs[1].plot(x, y, 'yo')  # Add yellow dot

            # Update the status text
            if self.selecting_source:
                self.status_text.set_text('Select a point on the right image.')
                self.status_text.set_color('black')
            else:
                self.status_text.set_text('Select a point on the left image.')
                self.status_text.set_color('black')

            # Update the canvas
            self.fig.canvas.draw()

            # Toggle between source and destination
            self.selecting_source = not self.selecting_source

    def show(self):
        plt.show()


img1 = plt.imread('Putin.jpg')
img2 = plt.imread('mandrill.png')
h1, w1, _ = img1.shape
img2 = cv2.resize(img2, (w1, h1))  # resize img2 to the same size as img1

selector = ControlPointsSelector(img1, img2)
selector.show()

# After closing the plot window
x_s, y_s = selector.x_s, selector.y_s
x_d, y_d = selector.x_d, selector.y_d

# Append the four corners of the image
h1, w1, _ = img1.shape
x_s.extend([0, 0, w1, w1])
y_s.extend([0, h1, 0, h1])
x_d.extend([0, 0, w1, w1])
y_d.extend([0, h1, 0, h1])

# exchange x and y, as the image is transposed 
x_s, y_s = y_s, x_s
x_d, y_d = y_d, x_d

print(x_s, y_s, x_d, y_d)

new_img = back_trans(img1, x_d, y_d, x_s, y_s, 2)

plt.imshow(new_img)
save_img(new_img, 'putinreg2.png')
plt.show()