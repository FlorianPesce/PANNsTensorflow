import pytest
import tensorflow as tf

from lib.model.wavegram import ConvPreWavBlock, Wavegram
from lib.training.mixup import generate_lambda

@pytest.fixture(scope="module")
def prewav_block():
    return ConvPreWavBlock(out_channels=64, pool_size=4)

@pytest.fixture(scope="module")
def wavegram():
    return Wavegram()


def test_init_convprewavblock():
    prewav_block = ConvPreWavBlock(out_channels=64, pool_size=4)
    assert prewav_block.conv1.filters == 64
    assert prewav_block.pooling.pool_size[0] == 4


def test_call_convprewavblock(prewav_block):
    inputs = tf.random.normal((20, 10000, 1))
    outputs = prewav_block(inputs)
    assert outputs.shape == tf.TensorShape((20, 2500, 64))

# NOTE Nothing to test?
def test_init_wavegram():
    wavegram = Wavegram()
    assert True

def test_call_wavegram(wavegram):
    inputs = tf.random.normal((20, 32000))
    mixup_lambda = generate_lambda(1, 1, 20)
    outputs = wavegram(inputs, mixup_lambda)
    assert outputs.shape == tf.TensorShape((20, 32, 50, 64))


