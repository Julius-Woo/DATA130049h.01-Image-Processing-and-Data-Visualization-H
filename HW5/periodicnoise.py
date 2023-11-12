import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt
from scipy import ndimage

clean_image = np.array(Image.open('./brainimg.png').convert('L'))
img_size = clean_image.shape

# Create a sinusoidal pattern for the noise
x = np.linspace(0, 2 * np.pi, img_size[1])
y = np.linspace(0, 2 * np.pi, img_size[0])
X, Y = np.meshgrid(x, y)

# Multi-frequency sinusoidal noise
multi_freq_noise = 10 * np.sin(10 * X) + 40 * np.sin(20 * Y)  + 20 * np.cos(30 * X) + 60 * np.cos(40 * Y)

# Create diagonal line effect by skewing the noise
diagonal_noise = ndimage.rotate(
    multi_freq_noise, 30, reshape=False, mode='reflect')
diagonal_noise = diagonal_noise[:img_size[0], :img_size[1]]

noisy_image_complex = clean_image + diagonal_noise

noisy_image_clipped = np.clip(noisy_image_complex, 0, 255)

save_image = Image.fromarray(noisy_image_clipped.astype(np.uint8))
save_image.save('./noisy_image.png')


# Show the clean and noisy images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(clean_image, cmap='gray')
axes[0].set_title('Clean Image')
axes[0].axis('off')

axes[1].imshow(noisy_image_clipped, cmap='gray')
axes[1].set_title('Noisy Image (with Periodic Noise)')
axes[1].axis('off')

plt.show()
