import tensorflow as tf
import random

class DataLoader(object):

    def __init__(self, image_paths, mask_paths, image_extension, image_size, \
                 crop_size=None, channels=[3, 3], palette=None,  seed=None):
        """
        Initializes the data loader object
        Args:
            image_paths: List of paths of train images.
            mask_paths: List of paths of train masks (segmentation masks)
            image_extension: The file format of images, either 'jpeg' or 'png'.
            image_size: Tuple of (Height, Width), the final height 
                        to resize images to.
            crop_size: Tuple of (crop_height, crop_width), the size
                       to randomly crop out from the original image.
                       Only needed if augmentation is enabled.
                       Note: crop_size can be larger than image_size
                       since orignal image is first cropped and then
                       resized.
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
        self.image_extension = image_extension
        self.palette = palette
        self.image_size = image_size
        self.crop_size = crop_size
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

        image = tf.cond(cond_crop_image, lambda: tf.random_crop(
            image, [self.crop_size[0], self.crop_size[1], self.channels[0]], seed=self.seed), lambda: tf.identity(image))
        mask = tf.cond(cond_crop_mask, lambda: tf.random_crop(
            mask, [self.crop_size[0], self.crop_size[1], self.channels[1]], seed=self.seed), lambda: tf.identity(mask))


        return image, mask


    def _flip_left_right(self, image, mask):
        """
        Randomly flips image and mask left or right in accord.
        """
        image = tf.image.random_flip_left_right(image, seed=self.seed)
        mask = tf.image.random_flip_left_right(mask, seed=self.seed)

        return image, mask


    def _normalize_data(self, image, mask):
        """
        Normalize image and mask within range 0-1.
        """
        image = tf.cast(image, tf.float32)
        image = image / 255.0

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

        if self.image_extension == 'png':
            images = tf.image.decode_png(image_content, channels=self.channels[0])
            masks = tf.image.decode_png(mask_content, channels=self.channels[1])
        elif self.image_extension == 'jpeg':
            images = tf.image.decode_jpeg(image_content, channels=self.channels[0])
            masks = tf.image.decode_jpeg(mask_content, channels=self.channels[1])
        else:
            raise ValueError("Specified image extension is not supported,\
                              please use either jpeg or png images")

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
            batch_size: Number of images/masks in each batch returned.
            one_hot_encode: Boolean, whether to one hot encode the mask image or not.
                            Encoding will done according to the palette specified when
                            initializing the object.
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

        # Batch the data
        data = data.batch(batch_size)

        # Normalize
        data = data.map(self._normalize_data,
                        num_parallel_calls=num_threads).prefetch(buffer)

        if shuffle:
            data = data.shuffle(buffer)

        # Create iterator
        iterator = tf.data.Iterator.from_structure(
            data.output_types, data.output_shapes)

        # Next element Op
        next_element = iterator.get_next()

        # Data set init. op
        init_op = iterator.make_initializer(data)

        return next_element, init_op
