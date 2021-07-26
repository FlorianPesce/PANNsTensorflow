import tensorflow as tf
from tf_stft import Spectrogram, Logmel

class ConvBlock(tf.keras.Model):

    def __init__(self, out_channels):
        super(ConvBlock, self).__init__()
        self.zero_pad1 = tf.keras.layers.ZeroPadding2D(padding = (1, 1))
        self.conv1 = tf.keras.layers.Conv2D(filters = out_channels,
        kernel_size=(3, 3), strides=(1, 1),
        padding = 'valid', use_bias=False, kernel_initializer='glorot_uniform')
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.zero_pad2 = tf.keras.layers.ZeroPadding2D(padding = (1, 1))
        self.conv2 = tf.keras.layers.Conv2D(filters = out_channels,
        kernel_size=(3, 3), strides=(1, 1),
        padding = 'valid', use_bias=False, kernel_initializer='glorot_uniform')
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, pool_size=(2, 2), pool_type='avg'):

        x = inputs
        x = tf.keras.layers.ReLU()(self.bn1(self.conv1(x)))
        x = tf.keras.layers.ReLU()(self.bn2(self.conv2(x)))

        if pool_type == 'max':
            x = tf.keras.layers.MaxPool2D(pool_size = pool_size)(x)
        elif pool_type == 'avg':
            x = tf.keras.layers.AveragePooling2D(pool_size = pool_size)(x)
        elif pool_type == 'avg+max':
            x1 = tf.keras.layers.AveragePooling2D(pool_size = pool_size)(x)
            x2 = tf.keras.layers.MaxPool2D(pool_size = pool_size)(x)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x

class Cnn14(tf.keras.Model):
    
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):

        super(Cnn14, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectogram extractor
        self.spectogram_extractor = Spectrogram(n_fft=

