import pytest
import tensorflow as tf

from lib.training.mixup import do_mixup, generate_lambda

@pytest.fixture(scope="module")
def input():
    # input is of size (batch_size,)
    return tf.constant([[1., 2.],
                        [7., 3.],
                        [5., 6.],
                        [2., 9.]])

@pytest.fixture(scope="module")
def mixup_lambda():
    # lambda is used for mixup and is of size (batch_size,)
    return tf.constant([[0.1, 0.8],
                        [0.3, 0.2],
                        [0.5, 0.6],
                        [0.6, 0.9]])


def test_do_mixup(input, mixup_lambda):
    # input is of size (batch_size,)
    input = tf.constant([[1., 2.],
                        [7., 3.],
                        [5., 6.],
                        [2., 9.]])

    # lambda is used for mixup and is of size (batch_size,)
    mixup_lambda = tf.constant([[0.1, 0.8],
                        [0.3, 0.2],
                        [0.5, 0.6],
                        [0.6, 0.9]])

    # errors
    wrong_mixup_lambda = tf.constant([[0.1, 0.8],
                        [0.3, 0.2],
                        [0.5, 0.6],
                        [0.6, 0.9],
                        [0.5, 0.4]])
    with pytest.raises(ValueError):
        output = do_mixup(input, wrong_mixup_lambda)

    # normal usage
    output = do_mixup(input, mixup_lambda)

    assert output.shape[0] == input.shape[0]
    assert output.dtype is tf.float32

    expected_output = tf.constant([[1.9, 3.4],
                                   [2.8, 2.2],
                                   [6., 4.8],
                                   [3.2, 8.7]])

    max_err = tf.math.reduce_max(tf.abs(output - expected_output))

    assert max_err < 1e-6


def test_generate_lambda():
    alpha = 2.0
    beta = 1.0
    batch_size = 10

    output = generate_lambda(alpha, beta, batch_size)

    assert output.shape == [batch_size]
    assert output.dtype is tf.float32

    assert tf.reduce_all(tf.math.greater(output,
        tf.zeros(dtype=tf.float32, shape=output.shape)))
    assert tf.reduce_all(tf.math.greater(tf.ones(dtype=tf.float32,
        shape=output.shape), output))




    






