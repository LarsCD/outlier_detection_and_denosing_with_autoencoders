import matplotlib.pyplot as plt
from PIL.ImageOps import grayscale

from denoising_autoencoder import *
raw_image = load_random_image_from_folder('undistorted_images')

plt.imshow(raw_image)
plt.show()

noise_image = add_noise(raw_image)

plt.imshow(noise_image)
plt.show()

denoised_image = denoise_single_image('model_path/denoising_autoencoder.keras', noise_image)

plt.imshow(denoised_image)
plt.show()
