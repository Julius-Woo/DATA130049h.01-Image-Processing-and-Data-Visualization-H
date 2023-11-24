from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from kmeansimg import kmeans


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

class Gmmem():
    def __init__(self, img, k, paras):
        self.img = img  # Image to be segmented
        self.num = k  # Number of classes
        self.shape = img.shape  # Shape of the image
        # Parameters for the GMM, including the weights(\pi_k), means(\mu) and variances(\sigma^2)
        self.paras = paras

    def phi(self, mean, var, x):
        # Normal probability density function
        return np.exp(-(x - mean) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)

    def get_prob(self, x):
        # Calculate the posterior probability for each class and return the class with the highest probability
        prob = np.array([self.phi(self.paras[1][i], self.paras[2][i], x) for i in range(self.num)])
        posterior = (prob * self.paras[0]) / \
            (np.sum(prob * self.paras[0]) + 1e-8)  # avoid zero division
        return int(np.argmax(posterior))

    def inference(self):
        # Perform inference on the image, and return the mask with the same shape as the image
        mask = np.array([self.get_prob(i) for i in self.img.reshape(-1)]
                        ).reshape(self.shape[0], self.shape[1])
        return mask

    def segmentation(self, max_iter, tol):
        # Perform segmentation on the image, alternatively perform E-step and M-step
        m = 0  # Number of iterations
        while True:
            P = self.E_step()  # E-step: Calculate posterior probabilities
            # M-step: Update model parameters
            new_paras = self.M_update_paras(P)
            if (abs(new_paras - self.paras) < tol).all():
                print('gmm Converged! Number of iteration: ', m)
                break
            elif m > max_iter:
                break
            else:
                self.paras = new_paras
                m += 1
        mask = self.inference()
        return mask

    def P_cal(self, x):
        # Calculate Q/posterior for every pixel
        prob = np.array([self.phi(self.paras[1][i], self.paras[2][i], x)
                        for i in range(self.num)])
        posterior = (prob * self.paras[0]) / \
            (np.sum(prob * self.paras[0]) + 1e-8)  # avoid zero division
        return posterior

    def E_step(self):
        # Calculate the posterior probabilities for each pixel in the image
        P = np.stack([self.P_cal(i)
                    for i in self.img.reshape(-1)], 0).T  # (k, h*w)
        return P

    def M_update_paras(self, P):
        # Update the model parameters based on the posterior probabilities
        new_paras = np.zeros(self.paras.shape)
        new_paras[1] = np.array(
            [np.sum(P[i] * self.img.reshape(-1)) / np.sum(P[i]) for i in range(self.num)])
        new_paras[2] = np.array([np.sum(P[i] * (self.img.reshape(-1) - new_paras[1][i]) ** 2) / np.sum(P[i])
                                for i in range(self.num)])
        new_paras[0] = np.sum(P, -1) / np.sum(P)
        new_paras[2][new_paras[2] < 1] = np.random.randint(1, 10, 1)  # Avoid zero variance
        return new_paras
    
    def create_newimg(self, max_iter=50, tol=1):
        # Create a new image based on the labels
        labels = self.segmentation(max_iter, tol)
        new_img = np.zeros(self.shape, dtype=np.uint8)
        for i in range(self.num):
            new_img[labels == i] = self.paras[1][i]
        return new_img
    
def initialize_paras(img, k):
    paras = np.zeros((3, k))
    paras[0] = np.ones(k) / k  # equal weights
    labels, centroids, _ = kmeans(img, k)
    paras[1] = centroids
    paras[2] = np.array([np.var(img[labels == i]) for i in range(k)])
    return paras


# # test
# # binarize the image
# k = 2
# img = img_read('./heart_noised.png')
# gmm = Gmmem(img, k, initialize_paras(img, k))
# img_new = gmm.create_newimg()
# show_img_gray(img_new)
# # save_img(img_new, './heart_gmm2.png')

# # more than 2 classes
# k = 3
# img = img_read('./heart_noised.png')
# height, width = img.shape
# gmm = Gmmem(img, k, initialize_paras(img, k))
# labels = gmm.segmentation(50,1)
# # Create a colored image based on labels
# img_new = np.zeros((height, width, 3), dtype=np.uint8)
# colors = np.array([[228, 143, 18], [4, 135, 158], [129, 254, 177]])
# for i in range(k):
#     img_new[labels == i] = colors[i]
# show_img_colour(img_new)
# # save_img(img_new, './heart_gmm3.png')

# more than 2 classes
k = 5
img = img_read('./HW6/heart_noised.png')
gmm = Gmmem(img, k, initialize_paras(img, k))
img_new = gmm.create_newimg()
show_img_gray(img_new)
# save_img(img_new, './heart_gmm5.png')