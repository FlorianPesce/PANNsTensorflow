import tensorflow as tf

def do_mixup(x, mixup_lambda):
    """
    Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes (1, 3, 5, ...).

    Parameters
    ----------
    x : (batch_size * 2, ...)
    mixup_lambda : [batch_size * 2,)

    Returns
    -------
    (batch_size, ...)
        
    """

    x_shape_tuple = x.get_shape()
    x_shape_list = x_shape_tuple.as_list()
    batch_size = x_shape_list[0] / 2
    list_perm = [k for k in range(batch_size)]
    list_perm[0] = list_perm[-1]
    list_perm[-1] = 0
    list_perm2 = [k for k in range(2*batch_size)]
    list_perm2[0] = list_perm[-1]
    list_perm2[-1] = 0

    if (batch_size != int(batch_size)):
        raise ValueError("Error: batch_size is not an integer. Make sure\
        that x has first dimension equal to batch_size * 2")

    out = tf.transpose((tf.transpose(x[0 :: 2], perm = list_perm)
        * mixup_lambda[0 :: 2] + \
        tf.transpose(x[1 :: 2], list_perm) * mixup_lambda[1 :: 2]),\
        perm= list_perm2)

    return out