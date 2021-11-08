from lib.training.mixup import generate_lambda
import tensorflow as tf
from lib.model.wavegram import Wavegram
from lib.model.logmel_spectrogram import LogmelSpectrogram
from lib.model.backbone_cnn import Cnn14, ConvBlock

class Wavegram_Logmel_Cnn14(tf.keras.Model):
    """
    Wavegram-Logmel-CNN architecture.
    """
    def __init__(self, sample_rate=32000, window_size=1024, hop_size=320,
                 mel_bins=64, fmin=50, fmax=14000,
                 classes_num=527, dropout_rate=0.2, include_top=True,
                 wavegram=None, backbone=None,
                 alpha_mixup=1., beta_mixup=1.,
                 name="wavegram_logmel_cnn14"):
        """
        Parameters - default values for Audioset
        ----------
        sample_rate : float, optional
           sampling rate of the incoming signal
        window_size : int, optional
           number of FFT components
        hop_size : int, optional
           number of audio samples between adjacent STFT columns in spectrogram
        mel_bins : int, optional
            number of Mel bands to generate
        fmin : float, optional
            lowest frequency in Hz
        fmax : float, optional
            highest frequency in Hz
        classes_num : int, optional
            number of output classes
        dropout_rate : float, optional
            dropout used in backbone, by default 0.2
        include_top : bool, optional
            whether to include the clipwise_output or not, by default True
        alpha_mixup : float, optional
            first beta distribution parameter for lambda mixup generation
        beta_mixup : float, optional
            second beta distribution parameter for lambda mixup generation
        """
        super(Wavegram_Logmel_Cnn14, self).__init__()

        self.wavegram = Wavegram() if wavegram is None else wavegram
        self.spectrogram = LogmelSpectrogram(sample_rate, window_size,
                hop_size, mel_bins, fmin, fmax)
        self.conv_block1 = ConvBlock(out_channels=64, pool_size=(2, 2))
        self.backbone = (Cnn14(dropout_rate, include_top) 
                if backbone is None else backbone)
        
        self.classes_num = classes_num
        self.include_top = include_top
        # if include_top is False, then fc_audioset is not needed
        if include_top:
            self.fc_audioset = tf.keras.layers.Dense(classes_num,
                use_bias=True, kernel_initializer='glorot_uniform')
 
        self.alpha_mixup = alpha_mixup
        self.beta_mixup = beta_mixup


    def call(self, inputs, training=True):
        """
        Parameters
        ----------
        inputs : (batch_size, data_length)
        training : bool, optional
        """

        if training:
            mixup_lambda = generate_lambda(self.alpha_mixup,
                self.beta_mixup, tf.shape(inputs)[0])
        else:
            mixup_lambda = tf.ones(shape=tf.shape(inputs))

        x1 = inputs
        x1 = self.wavegram(x1, mixup_lambda, training)

        x2 = inputs
        x2 = self.spectrogram(x2, mixup_lambda, training)
        x2 = self.conv_block1(x2)

        x = tf.concat([x1, x2], axis=1)

        embedding = self.backbone(x)

        if self.include_top:
            clipwise_output = tf.math.sigmoid(self.fc_audioset(embedding))
            return clipwise_output, embedding
        else:
            return embedding

        

        


