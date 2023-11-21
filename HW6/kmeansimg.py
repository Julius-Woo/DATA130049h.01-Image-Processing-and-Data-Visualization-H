from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def img_read(img_path):
    img = Image.open(img_path).convert('L')
    img = np.array(img)
    return img


def save_img(img, save_path):
    img = Image.fromarray(img)
    img.save(save_path)


def show_img_gray(img):
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.show()


def show_img_colour(img):
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.show()


def kmeans(img, k, max_iter=100, tol=1e-5):
    '''
    Use k-means algorithm to segment the image.
    '''
    # Initialize the centroids by selecting k random pixels from the image
    np.random.seed(7)  # for reproducibility
    centroids = img.ravel()[np.random.choice(img.size, k, replace=False)]

    height, width = img.shape
    labels = np.zeros((height, width), dtype=np.int_)

    m = 0  # Number of iterations
    for _ in range(max_iter):
        # Update the labels
        distances = (img[..., None] - centroids)**2  # broadcasting to subtract centroids from all pixels
        labels = np.argmin(distances, axis=2)  # find the index of the closest centroid

        # Update the centroids
        centroids_new = np.array(
            [img[labels == i].mean() if np.any(labels == i) else 0 for i in range(k)])  # avoid empty clusters

        # Check the convergence
        if np.sqrt(np.sum((centroids_new - centroids)**2)) < tol:
            print('Kmeans Converged! Number of iteration: ', m)
            break
        else:
            centroids = centroids_new
            m += 1

    new_img = np.zeros((height, width), dtype=np.uint8)
    for i in range(k):
        new_img[labels == i] = centroids[i]
    
    # if k > 2:
    #     # Create a colored image based on labels
    #     new_img = np.zeros((height, width, 3), dtype=np.uint8)
    #     colors = np.random.randint(0, 256, size=(k, 3), dtype=np.uint8)

    #     for i in range(k):
    #         new_img[labels == i] = colors[i]
    # else:
    #     # Create a black and white image based on labels
    #     new_img = np.zeros((height, width), dtype=np.uint8)
    #     colors = [255, 0]
    #     for i in range(k):
    #         new_img[labels == i] = colors[i]

    return labels, centroids, new_img

# # test
# # binarize the image
# k=2
# img = img_read('./heart_noised.png')
# img_new = kmeans(img, k)[2]
# show_img_gray(img_new)
# # save_img(img_new, './heart_kmeans2.png')

# # # more than 2 classes
# k=3
# img = img_read('./heart_noised.png')
# img_new1 = kmeans(img, k)[2]
# show_img_colour(img_new1)
# # save_img(img_new1, './heart_kmeans5.png')

# # more than 2 classes
# k=5
# img = img_read('./heart_noised.png')
# img_new1 = kmeans(img, k)[2]
# show_img_gray(img_new1)
# # save_img(img_new1, './heart_kmeans5.png')
