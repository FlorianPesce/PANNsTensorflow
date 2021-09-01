import pytest
import tensorflow as tf

from lib.training.mixup import do_mixup

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
    # do_mixup output
    output = do_mixup(input, mixup_lambda)

    assert output.shape[0] == input.shape[0]
    assert output.dtype == tf.float32

    expected_output = tf.constant([[1.9, 3.4],
                                   [2.8, 2.2],
                                   [6., 4.8],
                                   [3.2, 8.7]])

    bool_err = (tf.abs(output - expected_output)
           < tf.constant(1e-6, dtype=tf.float32, shape = input.shape))

    assert tf.reduce_all(bool_err)


    






