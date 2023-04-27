import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.preprocessing.image import load_img
import numpy as np

# Load the trained generator model
MODEL_PATH ='Models/9/9_g_model.h5'
generator = load_model(MODEL_PATH, custom_objects={'InstanceNormalization': InstanceNormalization})

# Generate a set of latent vectors using the training images
img = load_img('Dataset/CUHK/Training sketch/M2-043-01-sz1.jpg', target_size=(256, 256))
img = img_to_array(img)
img = np.expand_dims(img, axis=0)
for image in os.listdir('Dataset/CUHK/Training sketch'):
    path1 = os.path.join('Dataset/CUHK/Training sketch',image)
    img1 = load_img(path1, target_size=(256, 256))
    img1 = img_to_array(img1)
    img1 = np.expand_dims(img1, axis=0)
    img = np.append(img, img1, axis=0)

norm_img = (img.copy() - 127.5) / 127.5
latent_vectors = norm_img

# Generate fake images from the latent vectors using the generator
fake_images = generator.predict(latent_vectors)

# Reduce the dimensionality of the latent vectors using t-SNE
fake_images_2d = fake_images.reshape((fake_images.shape[0], -1))
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
latent_vectors_2d = tsne.fit_transform(fake_images_2d)

# Visualize the latent vectors in a scatter plot
plt.scatter(latent_vectors_2d[:, 0], latent_vectors_2d[:, 1], c=np.arange(len(latent_vectors_2d)))
plt.colorbar()
plt.savefig('latent_space2.png')
