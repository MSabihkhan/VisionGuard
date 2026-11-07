import tensorflow as tf


def prepare(raw_bytes, img):
    image = tf.io.decode_image(raw_bytes, channels=3, expand_animations=False)
    return tf.image.resize(image, (img, img))
