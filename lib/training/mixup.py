import tensorflow as tf
        
def generate_lambda(alpha, beta, batch_size):
    """Generate beta distribution and get mixup random coefficients.

    Parameters
    ----------
    alpha : float
        first distribution parameter
    beta : float
        second distribution parameter
    batch_size : int

    Returns
    -------
    tf.Tensor(, shape=(batch_size), dtype=float32)
        mixup random coefficients, sampled from beta distribution.
    """
    x = tf.random.gamma([batch_size], alpha)
    y = tf.random.gamma([batch_size], beta)

    return x /(x + y)

def do_mixup(x, mixup_lambda):
    """
    Mixup x with a circular permutation of x using a lambda vector.

    Parameters
    ----------
    x : (batch_size,)
    mixup_lambda : (batch_size,)

    Returns
    -------
    (batch_size, ...)
        
    """
    
    shape_x = tf.shape(x)
    shape_lambda = tf.shape(mixup_lambda)

    if (shape_x[0] != shape_lambda[0]):
        raise ValueError("x and mixup_lambda should have the same first dimension."
        f"Currently, their first dimensions are {shape_x[0]} and {shape_lambda[0]}.")

    out = x * mixup_lambda + (1 - mixup_lambda) * tf.roll(x, shift=1, axis=0)

    return out





