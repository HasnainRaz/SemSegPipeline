import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator
import random

def augment_dataset(images, masks, crop_size, image_size, batch_size):
    '''
    Returns Augmented images with either: random_crop, random_brightness,
    random_contrast, random_saturation, random_flip vertically, random_flip
    horizontally, defined by the boolean tensor params
    Inputs:
        images: Batch of images, shape = [batch, width, height, channels]
        masks: Batch of mask images, shape = [batch, width, height, channels]
        params: Placeholder tensor to define which random perturbations to apply
                out of the 6 mentioned above, shape = [6]
        crop_size: Size to crop an image
        image_size: Final size of the images to return
        batch_size: Number of samples in a batch
    Returns:
        images: Augmented images with any combination of augmentations applied
        masks: Segmentation masks flipped/cropped in accord with their image
    '''

    # Get seed to sync all random functions
    seed = random.random()
    cond_crop_image = tf.cast(tf.random_uniform([], maxval=2, dtype=tf.int32, seed=seed), tf.bool)
    cond_crop_mask = tf.cast(tf.random_uniform([], maxval=2, dtype=tf.int32, seed=seed), tf.bool)
    cond_brightness = tf.cast(tf.random_uniform([], maxval=2, dtype=tf.int32), tf.bool)
    cond_contrast = tf.cast(tf.random_uniform([], maxval=2, dtype=tf.int32), tf.bool)
    cond_saturation = tf.cast(tf.random_uniform([], maxval=2, dtype=tf.int32), tf.bool)
    cond_flip_lr = tf.cast(tf.random_uniform([], maxval=2, dtype=tf.int32), tf.bool)
    cond_flip_ud = tf.cast(tf.random_uniform([], maxval=2, dtype=tf.int32), tf.bool)
    # Apply same cropping on image and mask with the help of seed,
    # if true in params placeholder tensor
    images = tf.cond(cond_crop_image, lambda: tf.random_crop(
        images, [batch_size, int(crop_size[0] * 0.85), int(crop_size[1] * 0.85), 3], seed=seed), lambda: tf.identity(images))
    masks = tf.cond(cond_crop_mask, lambda: tf.random_crop(
        masks, [batch_size, int(crop_size[0] * 0.85), int(crop_size[1] * 0.85), 1], seed=seed), lambda: tf.identity(masks))

    # Apply random brightness changes if True in params placeholder tensor
    images = tf.cond(cond_brightness, lambda: tf.image.random_brightness(
        images, 0.01), lambda: tf.identity(images))
        
    # Apply random contrast changes if True in params
    images = tf.cond(cond_contrast, lambda: tf.image.random_contrast(
        images, 0.2, 1.8), lambda: tf.identity(images))

    # Saturation doesn't work on image batches, only single images,
    # Use map_fn to apply on batch
    def sat(x): return tf.image.random_saturation(x, 0.2, 1.8)
    images = tf.cond(cond_saturation, lambda: tf.map_fn(
        sat, images), lambda: tf.identity(images))

    # Define tf.identity equivalent which returns two inputs unchanged
    def unchanged(images, masks):
        return images, masks

    # Apply random horizontal flip if active in params,
    # tf.image.random_flip_left_right doesnt work on image batches
    def flip_l_r(images, masks):
        image, mask = [], []
        for i in range(batch_size):
            seed = random.random()
            image.append(tf.image.random_flip_left_right(
                images[i, :, :, :], seed=seed))
            mask.append(tf.image.random_flip_left_right(
                masks[i, :, :, :], seed=seed))
        images = tf.stack(image)
        masks = tf.stack(mask)
        return images, masks

    # Apply horizontal flip
    images, masks = tf.cond(cond_flip_lr, lambda: flip_l_r(
        images, masks), lambda: unchanged(images, masks))

    # Apply random vertical flip if active in params,
    # tf.image.random_flip_up_down doesnt work on image batches
    def flip_u_d(images, masks):
        image, mask = [], []
        for i in range(batch_size):
            seed = random.random()
            image.append(tf.image.random_flip_up_down(
                images[i, :, :, :], seed=seed))
            mask.append(tf.image.random_flip_up_down(
                masks[i, :, :, :], seed=seed))
        images = tf.stack(image)
        masks = tf.stack(mask)
        return images, masks

    # Apply the vertical flip
    images, masks = tf.cond(cond_flip_ud, lambda: flip_l_r(
        images, masks), lambda: unchanged(images, masks))

    # Finally resize the images to given width and height
    images = tf.image.resize_bicubic(images, image_size)
    masks = tf.image.resize_bicubic(masks, image_size)

    return images, masks


def _normalize_data(image, mask):
    '''Returns image and mask divided by 255 (vals in range 0.0 - 1.0)'''
    image = tf.cast(image, tf.float32)
    image = image / 255.0

    mask = tf.cast(mask, tf.float32)
    mask = mask / 255.0
    mask = tf.cast(mask, tf.int32)

    return image, mask


def _parse_data(image_paths, mask_paths):
    '''Reads image and mask files'''
    image_content = tf.read_file(image_paths)
    mask_content = tf.read_file(mask_paths)

    images = tf.image.decode_png(image_content, channels=3)
    masks = tf.image.decode_png(mask_content, channels=1)

    return images, masks


def data_batch(image_paths, mask_paths, batch_size=4, num_threads=2):
    '''Reads data, normalizes it, shuffles it, then batches it, returns a
       the next element in dataset op and the dataset initializer op.
       Inputs:
        image_paths: A list of paths to individual images
        mask_paths: A list of paths to individual mask images
        batch_size: Number of images/masks in each batch returned
        num_threads: Number of parallel calls to make
       Returns:
        next_element: A tensor with shape [2], where next_element[0]
                      is image batch, next_element[1] is the corresponding
                      mask batch
        init_op: Data initializer op, needs to be executed in a session
                 for the data queue to be filled up and the next_element op
                 to yield batches'''

    # Convert lists of paths to tensors for tensorflow
    images_name_tensor = tf.constant(image_paths)
    mask_name_tensor = tf.constant(mask_paths)

    # Create dataset out of the 2 files:
    data = Dataset.from_tensor_slices(
        (images_name_tensor, mask_name_tensor))

    # Parse images and labels
    data = data.map(
        _parse_data, num_threads=num_threads, output_buffer_size=6 * batch_size)

    # Normalize to be in range 0-1
    data = data.map(
        _normalize_data, num_threads=num_threads, output_buffer_size=6 * batch_size)

    # Shuffle the data queue
    data = data.shuffle(batch_size)

    # Create a batch of data
    data = data.batch(batch_size)

    # Create iterator
    iterator = Iterator.from_structure(data.output_types, data.output_shapes)

    # Next element Op
    next_element = iterator.get_next()

    # Data set init. op
    init_op = iterator.make_initializer(data)

    return next_element, init_op
