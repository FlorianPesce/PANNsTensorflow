import tensorflow as tf
from tf_stft import Spectrogram, Logmel
from tensorflow_utils import do_mixup

class ConvBlock(tf.keras.Model):
    """
    Convolutional Block Class.
    """
    def __init__(self, out_channels):
        """
        Parameters
        ----------
        out_channels : int
            Number of output channels
        """
        super(ConvBlock, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(filters = out_channels,
                                            kernel_size=3, strides=1,
                                            padding = 'same',
                                            use_bias=False,
                                            kernel_initializer='glorot_uniform')
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(filters = out_channels,
                                            kernel_size=3, strides=1,
                                            padding = 'same',
                                            use_bias=False,
                                            kernel_initializer='glorot_uniform')
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, pool_size=(2, 2), pool_type='avg'):

        # NOTE move pool_type to init.
        x = inputs
        x = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        x = tf.keras.activations.relu(self.bn2(self.conv2(x)))

        if pool_type == 'max':
            x = tf.keras.layers.MaxPool2D(pool_size = pool_size)(x)
        elif pool_type == 'avg':
            x = tf.keras.layers.AveragePooling2D(pool_size = pool_size)(x)
        elif pool_type == 'avg+max':
            x1 = tf.keras.layers.AveragePooling2D(pool_size = pool_size)(x)
            x2 = tf.keras.layers.MaxPool2D(pool_size = pool_size)(x)
            x = x1 + x2
        else:
            raise ValueError("pool_type should be one of the following:\
            max, avg or avg+max. Here, we got {}.".format(pool_type))
            # NOTE change to fstring
        
        return x

class Cnn14(tf.keras.Model):
    """
    CNN14 Backbone
    """
    # NOTE: I did everything. only leave backbone in here
    # NOTE add name argument in init
    def __init__(self, sample_rate, window_size, hop_size, mel_bins,
                 fmin, fmax, classes_num):
        # NOTE Add Docstring
        super(Cnn14, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size,
                                                 hop_length=hop_size) # NOTE Missing parameters: win_length, window, center, pad_mode, freeze_parameters

        self.logmel_extractor = Logmel(sample_rate=sample_rate,
                win_length=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref,
                amin=amin, top_db=top_db) # NOTE Missing parameter: freeze_parameters

        self.spec_augmenter = # NOTE Missing SpecAugmentation function

        self.bn0 = tf.keras.layers.BatchNormalization()

        self.conv_block1 = ConvBlock(out_channels=64)
        self.conv_block2 = ConvBlock(out_channels=128)
        self.conv_block3 = ConvBlock(out_channels=256)
        self.conv_block4 = ConvBlock(out_channels=512)
        self.conv_block5 = ConvBlock(out_channels=1024)
        self.conv_block6 = ConvBlock(out_channels=2048)

        # NOTE uuse_bias==True
        self.fc1 = tf.keras.layers.Dense(2048, use_bias=True)
        self.fc_audioset = tf.keras.layers.Dense(classes_num, use_bias=True)

        # NOTE Question: Need to initialize?. -> Do it in arguments.

    def call(self, inputs, mixup_lambda=None):
        # NOTE add training in call
        """
        Parameters
        ----------
        inputs : (batch_size, data_length)
        mixup_lambda : (batch_size * 2,), optional
        """

        # NOTE add comment to say that second dimension is channels
        x = self.spectrogram_extractor(inputs) # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x) # (batch_size, 1, time_steps, mel_bins)

        # NOTE investigate or ask qiuqiang. 
        x = tf.transpose(x, perm=[0, 3, 2, 1])
        x = self.bn0(x)
        x = tf.transpose(x, perm=[0, 3, 2, 1])

        if self.training:
            x = self.spec_augmenter(x)

        # NOTE move mixup bool as an attribut
        # NOTE create lambda uniform in call
        # NOTE create a lambda attribut: update it every time a forward function is used
        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        # NOTE add dropout_rates in init.
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = tf.keras.layers.Dropout(.2)(x) # NOTE add training attribute on dropout layers
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = tf.keras.layers.Dropout(.2)(x)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = tf.keras.layers.Dropout(.2)(x)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = tf.keras.layers.Dropout(.2)(x)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = tf.keras.layers.Dropout(.2)(x)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = tf.keras.layers.Dropout(.2)(x)
        x = tf.math.reduce_mean(x, axis=-1)

        # NOTE test if I need parenthesis
        (x1, _) = tf.math.reduce_max(x, axis=-1)
        x2 = tf.math.reduce_mean(x, axis=-1)
        x = x1 + x2
        x = tf.keras.layers.Dropout(.5)(x)
        x = tf.keras.activations.relu(self.fc1(x))
        embedding = tf.keras.layers.Dropout(.5)(x)
        clipwise_output = tf.math.sigmoid(self.fc_audioset(x))
        
        output_dict = {'clipwise_output': clipwise_output,
                       'embedding': embedding}

        return output_dict



