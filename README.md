# SemSeg Pipeline
This is an efficient semantic segmentaiton data input pipeline function for Tensorflow 2.

The pipeline is made to be multithreaded, and uses the tf data API to prerocess the image and segmentation masks with augmentations on the CPU.

# Augmentation Examples:
Following shows the same image, loaded with the pipeline, note the different augmentations (birghtness, contrast, saturation, cropping, flips) where the masks are changed accordingly. The example image is taken from the PASCAL VOC dataset.

<p align="center">
<img width="728" alt="screenshot 2018-10-16 at 22 56 50" src="https://user-images.githubusercontent.com/4294680/68091937-2e74f380-fe86-11e9-9d4d-87cf1b81a4ea.gif">
</p>


# Example use:

```python 
from dataloader import DataLoader
import tensorflow as tf
import os

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
                     image_size=[256, 256],
                     crop_percent=0.8,
                     channels=[3, 3],
                     seed=47)

# Parse the images and masks, and return the data in batches, augmented optionally.
dataset = dataset.data_batch(batch_size=BATCH_SIZE
                             augment=True, 
                             shuffle=True)

# Initialize the data queue
for image, mask in dataset:      
  # Do whatever you want now, like creating a feed dict and train your models,
  # You can also directly feed in the tf tensors to your models to avoid using a feed dict.

```

# Note
You can optionally also specify a color palette for your masks. If your masks are encoded as RGB images and you want to one hot encode them.
