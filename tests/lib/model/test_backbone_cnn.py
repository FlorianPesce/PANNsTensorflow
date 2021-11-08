import pytest
import tensorflow as tf
from tensorflow.python.framework.tensor_shape import TensorShape

from lib.model.backbone_cnn import ConvBlock, Cnn14

@pytest.fixture(scope="module")
def conv_block_avg():
    return ConvBlock(out_channels=64)

@pytest.fixture(scope="module")
def conv_block_max():
    return ConvBlock(out_channels=64, pool_type='max')

@pytest.fixture(scope="module")
def conv_block_avgmax():
    return ConvBlock(out_channels=64, pool_type='avg+max')

@pytest.fixture(scope="module")
def cnn14():
    return Cnn14()


def test_init_convblock():
    conv_block = ConvBlock(out_channels=64)
    assert conv_block.out_channels == 64
    assert conv_block.pool_size == (2, 2)
    assert conv_block.pool_type == 'avg'
    with pytest.raises(ValueError):
        conv_block = ConvBlock(out_channels=64, pool_type='mean')


def test_call_convblock(conv_block_avg, conv_block_max,
                        conv_block_avgmax):
    inputs = tf.random.normal((20, 64, 101, 1))
    outputs = conv_block_avg(inputs)
    assert outputs.shape == TensorShape([20, 32, 50, 64])
    outputs = conv_block_max(inputs)
    assert outputs.shape == TensorShape([20, 32, 50, 64])
    outputs = conv_block_avgmax(inputs)
    assert outputs.shape == TensorShape([20, 32, 50, 64])


def test_init_cnn14():
    cnn14 = Cnn14()
    assert cnn14.dropout_rate == 0.2


def test_call_cnn14(cnn14):
    inputs = tf.random.normal((20, 32, 50, 128))
    outputs = cnn14(inputs)
    print(outputs.shape)
    assert outputs.shape == TensorShape([20, 2048])


