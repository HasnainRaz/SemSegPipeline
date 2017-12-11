import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator
import random


def _resize_image(image, mask):
    image = tf.image.resize_bicubic(image, [480, 640], True)
    mask = tf.image.resize_bicubic(mask, [480, 640], True)
    return image, mask


def _corrupt_contrast(image, mask):
    image = tf.image.random_contrast(image, 0, 5)
    return image, mask


def _corrupt_saturation(image, mask):
    image = tf.image.random_saturation(image, 0, 5)
    return image, mask


def _corrupt_brightness(image, mask):
    image = tf.image.random_brightness(image, 5)
    return image, mask


def _random_crop(image, mask):
    seed = random.random()
    image = tf.random_crop(image, [240, 320, 3], seed=seed)
    mask = tf.random_crop(mask, [240, 320, 1], seed=seed)
    return image, mask


def _flip_image_horizontally(image, mask):
    seed = random.random()
    image = tf.image.random_flip_left_right(image, seed=seed)
    mask = tf.image.random_flip_left_right(mask, seed=seed)

    return image, mask


def _flip_image_vertically(image, mask):
    seed = random.random()
    image = tf.image.random_flip_up_down(image, seed=seed)
    mask = tf.image.random_flip_up_down(mask, seed=seed)

    return image, mask


def _normalize_data(image, mask):
    image = tf.cast(image, tf.float32)
    image = image / 255.0

    mask = tf.cast(mask, tf.float32)
    mask = mask / 255.0

    return image, mask


def _parse_data(image_paths, mask_paths):
    image_content = tf.read_file(image_paths)
    mask_content = tf.read_file(mask_paths)

    images = tf.image.decode_png(image_content, channels=3)
    masks = tf.image.decode_png(mask_content, channels=1)

    return images, masks


def data_batch(image_paths, mask_paths, params, batch_size=4, num_threads=2):
    # Convert lists of paths to tensors for tensorflow
    images_name_tensor = tf.constant(image_paths)
    mask_name_tensor = tf.constant(mask_paths)

    # Create dataset out of the 3 files:
    data = Dataset.from_tensor_slices(
        (images_name_tensor, mask_name_tensor))

    # Parse images and labels
    data = data.map(
        _parse_data, num_threads=num_threads, output_buffer_size=6 * batch_size)

    # Concat. band 1, band 2 and composite images along the depth
    data = data.map(
        _normalize_data, num_threads=num_threads, output_buffer_size=6 * batch_size)

    if params['crop'] and not random.randint(0, 1):
        data = data.map(_random_crop, num_threads=num_threads,
                        output_buffer_size=6 * batch_size)

    if params['brightness'] and not random.randint(0, 1):
        data = data.map(_corrupt_brightness, num_threads=num_threads,
                        output_buffer_size=6 * batch_size)

    if params['contrast'] and not random.randint(0, 1):
        data = data.map(_corrupt_contrast, num_threads=num_threads,
                        output_buffer_size=6 * batch_size)

    if params['saturation'] and not random.randint(0, 1):
        data = data.map(_corrupt_saturation, num_threads=num_threads,
                        output_buffer_size=6 * batch_size)

    if params['flip_horizontally'] and not random.randint(0, 1):
        data = data.map(_flip_image_horizontally,
                        num_threads=num_threads, output_buffer_size=6 * batch_size)

    if params['flip_vertically'] and not random.randint(0, 1):
        data = data.map(_flip_image_vertically, num_threads=num_threads,
                        output_buffer_size=6 * batch_size)

    # Shuffle the data queue
    data = data.shuffle(len(image_paths))

    # Create a batch of data
    data = data.batch(batch_size)

    data = data.map(_resize_image, num_threads=num_threads,
                    output_buffer_size=6 * batch_size)

    # Create iterator
    iterator = Iterator.from_structure(data.output_types, data.output_shapes)

    # Next element Op
    next_element = iterator.get_next()

    # Data set init. op
    init_op = iterator.make_initializer(data)

    return next_element, init_op
