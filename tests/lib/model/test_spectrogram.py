import pytest
import tensorflow as tf
from tensorflow.python.framework.tensor_shape import TensorShape

from lib.model.logmel_spectrogram import LogmelSpectrogram
from lib.training.mixup import generate_lambda

@pytest.fixture(scope="module")
def spectrogram():
    return LogmelSpectrogram(sample_rate=32000, window_size=1024,
    hop_size=320, mel_bins=64, fmin=50, fmax=14000)


def test_init_spectrogram():
    spectrogram = LogmelSpectrogram(sample_rate=32000, window_size=1024,
    hop_size=320, mel_bins=64, fmin=50, fmax=14000)
    assert spectrogram.sample_rate == 32000
    assert spectrogram.window_size == 1024
    assert spectrogram.hop_size == 320
    assert spectrogram.mel_bins == 64
    assert spectrogram.fmin == 50
    assert spectrogram.fmax == 14000


def test_call_spectrogram(spectrogram):
    inputs = tf.random.normal((20, 32000))
    mixup_lambda = generate_lambda(1, 1, 20)
    outputs = spectrogram(inputs, mixup_lambda)
    assert outputs.shape == TensorShape([20, 64, 101, 1])
