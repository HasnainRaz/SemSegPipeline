# Tensorflow-input-pipeline

This is an input pipeline function for Tensorflow, which uses the Dataset API, and is designed for use with semantic segmentation datasets.

I have observed that generally importing your own data into tensorflow for deep learning/machine learning problems is...well...a problem, this code aims to simplify that, and get you up and running with your deep learning projects. The code is simple and readable, so you can easily edit and extend it for your own projects.

# Note
This code file is meant as a guide for anyone stuck at functions for loading your own data into Tensorflow, generally most problems in ML will follow the skeleton of this example, where you load image and labels (here label is just another image) -> you will preprocess this loaded data -> batch it -> return an iterator over it.

# Example use:

```python 
import matplotlib.pyplot as plt
import tensorflow as tf
from dataloader import DataLoader
import os

plt.ioff()

IMAGE_DIR_PATH = 'data/training/images'
MASK_DIR_PATH = 'data/training/masks'

# create list of PATHS
image_paths = [os.path.join(IMAGE_DIR_PATH, x) for x in os.listdir(IMAGE_DIR_PATH) if x.endswith('.png')]
mask_paths = [os.path.join(MASK_DIR_PATH, x) for x in os.listdir(MASK_DIR_PATH) if x.endswith('.png')]

# Where image_paths[0] = 'data/training/images/image_0.png' 
# And mask_paths[0] = 'data/training/masks/mask_0.png'

# Initialize the dataloader object
dataset = DataLoader(image_paths=image_paths,
                     mask_paths=mask_paths,
                     image_extension='png',
                     image_size=[256, 256],
                     crop_size=[400, 400],
                     channels=[3, 3]
                     palette=[0, 255],
                     seed=47)

# Parse the images and masks, and return the data in batches, augmented optionally.
data, init_op = dataset.data_batch(augment=True, 
                                   shuffle=True,
                                   batch_size=BATCH_SIZE,
                                   num_threads=4,
                                   buffer=60)


with tf.Session() as sess:
  # Initialize the data queue
  sess.run(init_op)
  # Evaluate the tensors
  aug_image, aug_mask = sess.run(data)
                                 
  # Do whatever you want now, like creating a feed dict and train your models,
  # You can also directly feed in the tf tensors to your models to avoid using a feed dict.

```
Additional changes will be coming, in the meantime, any ideas are welcome.
