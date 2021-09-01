import tensorflow as tf

def do_mixup(x, mixup_lambda):
    """
    Mixup x with a circular permutation of x using a lambda vector

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


