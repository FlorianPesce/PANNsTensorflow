import tensorflow as tf
from backbone_cnn import ConvBlock
from lib.training.mixup import do_mixup

class ConvPreWavBlock(tf.keras.Model):
    """
    1-d Convolutional Block Class used before Wavegram
    """
    def __init__(self, out_channels, pool_size):
        """
        Parameters
        ----------
        out_channels : int
            Number of output channels
        pool_size : int
            maxpooling size
        """

        super(ConvPreWavBlock, self).__init__()

        self.conv1 = tf.keras.layers.Conv1D(filters=out_channels,
                                            kernel_size=3, strides=1,
                                            padding='same', use_bias=False,
                                            kernel_initializer='glorot_uniform')
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv1D(filters=out_channels,
                                            kernel_size=3, strides=1,
                                            padding='same', use_bias=False,
                                            dilation_rate=2,
                                            kernel_initializer='glorot_uniform')
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.pooling = tf.keras.layers.MaxPool1D(pool_size = pool_size)

    def call(self, inputs):

        x = inputs
        x = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        x = tf.keras.activations.relu(self.bn2(self.conv2(x)))
        x = self.pooling(x)

        return x
        
class Wavegram(tf.keras.Model):
    """
    Class for computing Wavegram.
    """
    def __init__(self):

        # NOTE add question for Qiuqiang: modularity
        super(Wavegram, self).__init__()

        self.pre_conv0_padding = tf.keras.layers.ZeroPadding1D(padding=5)
        self.pre_conv0 = tf.keras.layers.Conv1D(filters=64,
                                            kernel_size=11, strides=5,
                                            padding='valid', use_bias=False,
                                            kernel_initializer='glorot_uniform')
        self.pre_bn0 = tf.keras.layers.BatchNormalization()

        self.pre_block1 = ConvPreWavBlock(out_channels=64, pool_size=4)
        self.pre_block2 = ConvPreWavBlock(out_channels=128, pool_size=4)
        self.pre_block3 = ConvPreWavBlock(out_channels=128, pool_size=4)
        self.pre_block4 = ConvBlock(out_channels=64, pool_size=(2, 1))

    def call(self, inputs, mixup_lambda, training=True):
        """
        Parameters
        ----------
        inputs : (batch_size, data_length)
        mixup_lambda : (batch_size,)
        training : bool, optional
        """

        x = tf.expand_dims(inputs, axis=1)
        x = self.pre_conv0(self.pre_conv0_padding(x))
        x = tf.keras.activations.relu(self.pre_bn0(x))
        x = self.pre_block1(x)
        x = self.pre_block2(x)
        x = self.pre_block3(x)
        x = tf.reshape(x, (tf.shape(x)[0], -1, 32, tf.shape(x)[1]))
        x = tf.transpose(x, perm=[0,1,3,2])
        x = self.pre_block4(x)

        # Mixup on Wavegram
        if training:
            x = do_mixup(x, mixup_lambda)

        return x


