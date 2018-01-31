import tensorflow as tf
import random


def _corrupt_brightness(image, mask):
    """Radnomly applies a random brightness change."""
    cond_brightness = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32), tf.bool)
    image = tf.cond(cond_brightness, lambda: tf.image.random_hue(
        image, 0.1), lambda: tf.identity(image))
    return image, mask


def _corrupt_contrast(image, mask):
    """Randomly applies a random contrast change."""
    cond_contrast = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32), tf.bool)
    image = tf.cond(cond_contrast, lambda: tf.image.random_contrast(
        image, 0.2, 1.8), lambda: tf.identity(image))
    return image, mask


def _corrupt_saturation(image, mask):
    """Randomly applies a random saturation change."""
    cond_saturation = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32), tf.bool)
    image = tf.cond(cond_saturation, lambda: tf.image.random_saturation(
        image, 0.2, 1.8), lambda: tf.identity(image))
    return image, mask


def _crop_random(image, mask):
    """Randomly crops image and mask in accord."""
    seed = random.random()
    cond_crop_image = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32, seed=seed), tf.bool)
    cond_crop_mask = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32, seed=seed), tf.bool)

    image = tf.cond(cond_crop_image, lambda: tf.random_crop(
        image, [int(480 * 0.85), int(640 * 0.85), 3], seed=seed), lambda: tf.identity(image))
    mask = tf.cond(cond_crop_mask, lambda: tf.random_crop(
        mask, [int(480 * 0.85), int(640 * 0.85), 1], seed=seed), lambda: tf.identity(mask))
    image = tf.expand_dims(image, axis=0)
    mask = tf.expand_dims(mask, axis=0)

    image = tf.image.resize_images(image, [480, 640])
    mask = tf.image.resize_images(mask, [480, 640])

    image = tf.squeeze(image, axis=0)
    mask = tf.squeeze(mask, axis=0)

    return image, mask


def _flip_left_right(image, mask):
    """Randomly flips image and mask left or right in accord."""
    seed = random.random()
    image = tf.image.random_flip_left_right(image, seed=seed)
    mask = tf.image.random_flip_left_right(mask, seed=seed)

    return image, mask


def _normalize_data(image, mask):
    """Normalize image and mask within range 0-1."""
    image = tf.cast(image, tf.float32)
    image = image / 255.0

    mask = tf.cast(mask, tf.float32)
    mask = mask / 255.0
    mask = tf.cast(mask, tf.int32)

    return image, mask


def _resize_data(image, mask):
    """Resizes images to smaller dimensions."""
    image = tf.image.resize_images(image, [480, 640])
    mask = tf.image.resize_images(mask, [480, 640])

    return image, mask


def _parse_data(image_paths, mask_paths):
    """Reads image and mask files"""
    image_content = tf.read_file(image_paths)
    mask_content = tf.read_file(mask_paths)

    images = tf.image.decode_png(image_content, channels=3)
    masks = tf.image.decode_png(mask_content, channels=1)

    return images, masks


def data_batch(image_paths, mask_paths, augment=False, batch_size=4, num_threads=2):
    """Reads data, normalizes it, shuffles it, then batches it, returns a
       the next element in dataset op and the dataset initializer op.
       Inputs:
        image_paths: A list of paths to individual images
        mask_paths: A list of paths to individual mask images
        augment: Boolean, whether to augment data or not
        batch_size: Number of images/masks in each batch returned
        num_threads: Number of parallel calls to make
       Returns:
        next_element: A tensor with shape [2], where next_element[0]
                      is image batch, next_element[1] is the corresponding
                      mask batch
        init_op: Data initializer op, needs to be executed in a session
                 for the data queue to be filled up and the next_element op
                 to yield batches"""

    # Convert lists of paths to tensors for tensorflow
    images_name_tensor = tf.constant(image_paths)
    mask_name_tensor = tf.constant(mask_paths)

    # Create dataset out of the 2 files:
    data = tf.data.Dataset.from_tensor_slices(
        (images_name_tensor, mask_name_tensor))

    # Parse images and labels
    data = data.map(
        _parse_data, num_parallel_calls=num_threads).prefetch(30)

    # If augmentation is to be applied
    if augment:
        data = data.map(_corrupt_brightness,
                        num_parallel_calls=num_threads).prefetch(30)

        data = data.map(_corrupt_contrast,
                        num_parallel_calls=num_threads).prefetch(30)

        data = data.map(_corrupt_saturation,
                        num_parallel_calls=num_threads).prefetch(30)

        data = data.map(
            _crop_random, num_parallel_calls=num_threads).prefetch(30)

        data = data.map(_flip_left_right,
                        num_parallel_calls=num_threads).prefetch(30)

    # Batch the data
    data = data.batch(batch_size)

    # Resize to smaller dims for speed
    data = data.map(_resize_data, num_parallel_calls=num_threads).prefetch(30)

    # Normalize
    data = data.map(_normalize_data,
                    num_parallel_calls=num_threads).prefetch(30)

    data = data.shuffle(30)

    # Create iterator
    iterator = tf.data.Iterator.from_structure(
        data.output_types, data.output_shapes)

    # Next element Op
    next_element = iterator.get_next()

    # Data set init. op
    init_op = iterator.make_initializer(data)

    return next_element, init_op
