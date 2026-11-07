import numpy as np
import pytest
import tensorflow as tf

from preprocess import prepare


def test_prepare_resizes_to_square():
    image = np.zeros((50, 80, 3), dtype="uint8")
    out = prepare(tf.io.encode_png(image).numpy(), 64)
    assert out.shape == (64, 64, 3)


def test_prepare_keeps_pixel_range():
    image = np.full((30, 30, 3), 255, dtype="uint8")
    out = prepare(tf.io.encode_png(image).numpy(), 32)
    assert float(out.numpy().max()) <= 255.0


def test_prepare_rejects_garbage():
    with pytest.raises(tf.errors.InvalidArgumentError):
        prepare(b"garbage", 32)
