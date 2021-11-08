import pytest
import tensorflow as tf
from tensorflow.python.framework.tensor_shape import TensorShape

from lib.model.pann import Wavegram_Logmel_Cnn14

@pytest.fixture(scope="module")
def pann():
    return Wavegram_Logmel_Cnn14()

@pytest.fixture(scope="module")
def pann_without_top():
    return Wavegram_Logmel_Cnn14(include_top=False)


def test_init_pann():
    pann = Wavegram_Logmel_Cnn14()
    assert pann.spectrogram.sample_rate == 32000
    assert pann.spectrogram.window_size == 1024
    assert pann.spectrogram.hop_size == 320
    assert pann.spectrogram.mel_bins == 64
    assert pann.spectrogram.fmin == 50
    assert pann.spectrogram.fmax == 14000
    assert pann.classes_num == 527
    assert pann.backbone.dropout_rate == 0.2
    assert pann.include_top == True
    assert pann.alpha_mixup == 1.
    assert pann.beta_mixup == 1.
    pann_withouttop = Wavegram_Logmel_Cnn14(include_top=False)
    assert pann_withouttop.include_top == False


def test_call_pann(pann, pann_without_top):
    inputs = tf.random.normal((20,32000))
    clipwise_output, embedding = pann(inputs)
    assert embedding.shape == TensorShape([20, 2048]) 
    assert clipwise_output.shape == TensorShape([20, 527])
    embedding2 = pann_without_top(inputs, training=False)
    assert embedding2.shape == TensorShape([20, 2048])
