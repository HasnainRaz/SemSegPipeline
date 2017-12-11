# Tensorflow-input-pipeline

This is an example input pipeline function for Tensorflow, which uses the Dataset API.

I have observed that generally importing your own data into tensorflow for Deep learning/Machine learning problems is...well...a problem, this code aims to simplify that, and get you up and running with your deep learning projects, use the provided helper functions alongwith the necessary changes needed for your specific project.

The code currently is programmed to be used for **semantic segmentation** tasks, where the input is an image, and the label is a binary mask image.

# Example use:

```python 
import matplotlib.pyplot as plt
import utility

IMAGE_DIR_PATH = 'data/training/images'
MASK_DIR_PATH = 'data/training/masks'

# create list of PATHS
image_paths = [os.path.join(IMAGE_DIR_PATH, x) for x in os.listdir(IMAGE_DIR_PATH) if x.endswith('.png')]
mask_paths = [os.path.join(MASK_DIR_PATH, x) for x in os.listdir(MASK_DIR_PATH) if x.endswith('.png')]

# Where image_paths[0] = '/data/training/images/image_0.png' and mask_paths[0] = 'data/training/masks/image_0_mask.png'

# Specify the augmentations to carry out on the data-set
params = {'brightness': True, 'contrast': True,
          'crop': True, 'saturation': True,
          'flip_horizontally': True,
          'flip_vertically': True}

data, init_op = utility.data_batch(image_paths, mask_paths, params)
image_tensor, mask_tensor = data

with tf.Session() as sess:
  sess.run(init_op)
  # Evaluate the tensors
  image, mask = sess.run([image_tensor, mask_tensor])
  # Confirming everything is working by visualizing
  plt.imshow(image)
  plt.mask(mask)
  # Do whatever you want now, like creating a feed dict and train your models

```

Improvements, suggestions, mistakes and bug fixes are welcome.
