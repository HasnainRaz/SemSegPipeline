import tensorflow as tf
import random

class DataLoader(object):
    """A TensorFlow Dataset API based loader for semantic segmentation problems."""

    def __init__(self, image_paths, mask_paths, image_size, crop_percent=0.8, channels=[3, 3], palette=None, seed=None):
        """
        Initializes the data loader object
        Args:
            image_paths: List of paths of train images.
            mask_paths: List of paths of train masks (segmentation masks)
            image_size: Tuple of (Height, Width), the final height 
                        to resize images to.
            crop_percent: Float in the range 0-1, defining percentage of image to randomly
                          crop.
            channels: List of ints, first element is number of channels in images,
                      second is the number of channels in the mask image (needed to
                      correctly read the images into tensorflow.)
            palette: A list of pixel values in the mask, the index of a value
                     in palette becomes the channel index of the value in mask.
                     for example, all if mask is binary (0, 1), then palette should
                     be [0, 1], mask will then have depth 2, where the first index along depth
                     will be 1 where the original mask was 0, and the second index along depth will
                     be 1 where the original mask was 1. Works for and rgb palette as well,
                     specify the palette as: [[255,255,255], [0, 255, 255]] etc.
                     (one hot encoding).
            seed: An int, if not specified, chosen randomly. Used as the seed for the RNG in the 
                  data pipeline.

        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.palette = palette
        self.image_size = image_size
        self.crop_percent = tf.constant(crop_percent, tf.float32)
        self.channels = channels
        if seed is None:
            self.seed = random.randint(0, 1000)
        else:
            self.seed = seed

    def _corrupt_brightness(self, image, mask):
        """
        Radnomly applies a random brightness change.
        """
        cond_brightness = tf.cast(tf.random_uniform(
            [], maxval=2, dtype=tf.int32), tf.bool)
        image = tf.cond(cond_brightness, lambda: tf.image.random_hue(
            image, 0.1), lambda: tf.identity(image))
        return image, mask


    def _corrupt_contrast(self, image, mask):
        """
        Randomly applies a random contrast change.
        """
        cond_contrast = tf.cast(tf.random_uniform(
            [], maxval=2, dtype=tf.int32), tf.bool)
        image = tf.cond(cond_contrast, lambda: tf.image.random_contrast(
            image, 0.2, 1.8), lambda: tf.identity(image))
        return image, mask


    def _corrupt_saturation(self, image, mask):
        """
        Randomly applies a random saturation change.
        """
        cond_saturation = tf.cast(tf.random_uniform(
            [], maxval=2, dtype=tf.int32), tf.bool)
        image = tf.cond(cond_saturation, lambda: tf.image.random_saturation(
            image, 0.2, 1.8), lambda: tf.identity(image))
        return image, mask


    def _crop_random(self, image, mask):
        """
        Randomly crops image and mask in accord.
        """
        cond_crop_image = tf.cast(tf.random_uniform(
            [], maxval=2, dtype=tf.int32, seed=self.seed), tf.bool)
        cond_crop_mask = tf.cast(tf.random_uniform(
            [], maxval=2, dtype=tf.int32, seed=self.seed), tf.bool)


        shape = tf.cast(tf.shape(image), tf.float32)
        h = tf.cast(shape[0] * self.crop_percent, tf.int32)
        w = tf.cast(shape[1] * self.crop_percent, tf.int32)

        image = tf.cond(cond_crop_image, lambda: tf.random_crop(
            image, [h, w, self.channels[0]], seed=self.seed), lambda: tf.identity(image))
        mask = tf.cond(cond_crop_mask, lambda: tf.random_crop(
            mask, [h, w, self.channels[1]], seed=self.seed), lambda: tf.identity(mask))

        return image, mask


    def _flip_left_right(self, image, mask):
        """
        Randomly flips image and mask left or right in accord.
        """
        image = tf.image.random_flip_left_right(image, seed=self.seed)
        mask = tf.image.random_flip_left_right(mask, seed=self.seed)

        return image, mask


    def _resize_data(self, image, mask):
        """
        Resizes images to specified size.
        """
        image = tf.expand_dims(image, axis=0)
        mask = tf.expand_dims(mask, axis=0)

        image = tf.image.resize_images(image, self.image_size)
        mask = tf.image.resize_nearest_neighbor(mask, self.image_size)

        image = tf.squeeze(image, axis=0)
        mask = tf.squeeze(mask, axis=0)
        
        return image, mask


    def _parse_data(self, image_paths, mask_paths):
        """
        Reads image and mask files depending on
        specified exxtension.
        """
        image_content = tf.read_file(image_paths)
        mask_content = tf.read_file(mask_paths)

        images = tf.image.decode_image(image_content, channels=self.channels[0])
        masks = tf.image.decode_image(mask_content, channels=self.channels[1])

        return images, masks


    def _one_hot_encode(self, image, mask):
        """
        Converts mask to a one-hot encoding specified by the semantic map.
        """
        one_hot_map = []
        for colour in self.palette:
            class_map = tf.reduce_all(tf.equal(mask, colour), axis=-1)
            one_hot_map.append(class_map)
        one_hot_map = tf.stack(one_hot_map, axis=-1)
        one_hot_map = tf.cast(one_hot_map, tf.float32)
        
        return image, one_hot_map

    def data_batch(self, augment=False, shuffle=False, one_hot_encode=False, batch_size=4, num_threads=1, buffer=30):
        """
        Reads data, normalizes it, shuffles it, then batches it, returns a
        the next element in dataset op and the dataset initializer op.
        Inputs:
            augment: Boolean, whether to augment data or not.
            shuffle: Boolean, whether to shuffle data in buffer or not.
            one_hot_encode: Boolean, whether to one hot encode the mask image or not.
                            Encoding will done according to the palette specified when
                            initializing the object.
            batch_size: Number of images/masks in each batch returned.
            num_threads: Number of parallel subprocesses to load data.
            buffer: Number of images to prefetch in buffer.
        Returns:
            next_element: A tensor with shape [2], where next_element[0]
                          is image batch, next_element[1] is the corresponding
                          mask batch.
            init_op: Data initializer op, needs to be executed in a session
                     for the data queue to be filled up and the next_element op
                     to yield batches.
        """

        # Convert lists of paths to tensors for tensorflow
        images_name_tensor = tf.constant(self.image_paths)
        mask_name_tensor = tf.constant(self.mask_paths)

        # Create dataset out of the 2 files:
        data = tf.data.Dataset.from_tensor_slices(
            (images_name_tensor, mask_name_tensor))

        # Parse images and labels
        data = data.map(
            self._parse_data, num_parallel_calls=num_threads).prefetch(buffer)

        # If augmentation is to be applied
        if augment:
            data = data.map(self._corrupt_brightness,
                            num_parallel_calls=num_threads).prefetch(buffer)

            data = data.map(self._corrupt_contrast,
                            num_parallel_calls=num_threads).prefetch(buffer)

            data = data.map(self._corrupt_saturation,
                            num_parallel_calls=num_threads).prefetch(buffer)

            data = data.map(self._crop_random, num_parallel_calls=num_threads).prefetch(buffer)

            data = data.map(self._flip_left_right,
                            num_parallel_calls=num_threads).prefetch(buffer)

        # Resize to smaller dims for speed
        data = data.map(self._resize_data, num_parallel_calls=num_threads).prefetch(buffer)

        # One hot encode the mask
        if one_hot_encode:
            if self.palette is None:
                raise ValueError('No Palette for one-hot encoding specified in the data loader!')
            data = data.map(self._one_hot_encode, num_parallel_calls=num_threads).prefetch(buffer)

        if shuffle:
            data = data.shuffle(buffer)
        
        # Batch the data
        data = data.batch(batch_size)

        # Create iterator
        iterator = tf.data.Iterator.from_structure(
            data.output_types, data.output_shapes)

        # Next element Op
        next_element = iterator.get_next()

        # Data set init. op
        init_op = iterator.make_initializer(data)

        return next_element, init_op
