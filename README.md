# Tensorflow-input-pipeline

This is an example input pipeline function for Tensorflow, which uses the Dataset API.

I have observed that generally importing your own data into tensorflow for Deep learning/Machine learning problems is...well...a problem, this code aims to simplify that, and get you up and running with your deep learning projects, use the provided helper functions along with the necessary changes needed for your specific project.

The code currently is programmed to be used for **semantic segmentation** tasks, where the input is an image, and the label is a binary mask image. You would need to modify the '_parse_data' function in the code for your own examples, along with the augmentation functions.

# Example use:

```python 
import matplotlib.pyplot as plt
import tensorflow as tf
import utility
import os

IMAGE_DIR_PATH = 'data/training/images'
MASK_DIR_PATH = 'data/training/masks'
BATCH_SIZE = 4

plt.ioff()

# create list of PATHS
image_paths = [os.path.join(IMAGE_DIR_PATH, x) for x in os.listdir(IMAGE_DIR_PATH) if x.endswith('.png')]
mask_paths = [os.path.join(MASK_DIR_PATH, x) for x in os.listdir(MASK_DIR_PATH) if x.endswith('.png')]

# Where image_paths[0] = '/data/training/images/image_0.png' 
# And mask_paths[0] = 'data/training/masks/image_0_mask.png'

# Parse the images and masks, and return the data in batches, augmented optionally
data, init_op = utility.data_batch(image_paths, mask_paths, augment=True, batch_size=BATCH_SIZE)
# Get the image and mask op from the returned dataset
aug_image_tensor, aug_mask_tensor = data


with tf.Session() as sess:
  sess.run(init_op)
  # Evaluate the tensors
  aug_image, aug_mask = sess.run([aug_image_tensor, aug_mask_tensor])
                                 
  # Confirming everything is working by visualizing
  plt.figure('augmented image')
  plt.imshow(aug_image[0, :, :, :])
  plt.figure('augmented mask')
  plt.imshow(aug_mask[0, :, :])
  plt.show()
  # Do whatever you want now, like creating a feed dict and train your models

```
Additional changes will be coming, in the meantime, any ideas are welcome.
