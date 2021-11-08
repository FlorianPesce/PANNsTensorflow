import tensorflow as tf
from lib.model.tf_stft import Spectrogram, Logmel
from lib.training.mixup import do_mixup

class LogmelSpectrogram(tf.keras.layers.Layer):
    """
    Computes Log mel Spectrogram
    """
    def __init__(self, sample_rate, window_size, hop_size, mel_bins,
                 fmin, fmax, name="logmel_spectrogram"):
        """
        Parameters
        ----------
        sample_rate : float
           sampling rate of the incoming signal
        window_size : int
           number of FFT components
        hop_size : int, optional
           number of audio samples between adjacent STFT columns
        mel_bins : int
            number of Mel bands to generate
        fmin : float
            lowest frequency in Hz
        fmax : float
            highest frequency in Hz
        """
        
        super(LogmelSpectrogram, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.mel_bins = mel_bins
        self.fmin = fmin
        self.fmax = fmax

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(win_length=window_size,
                                                 hop_length=hop_size)
        # NOTE Missing parameters: win_length, window, center, pad_mode, freeze_parameters

        self.logmel_extractor = Logmel(sample_rate=sample_rate,
                win_length=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax,
                amin=amin, top_db=top_db)
        # NOTE Missing parameter: freeze_parameters, ref

        # self.spec_augmenter = # NOTE Missing SpecAugmentation function

        self.bn0 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, mixup_lambda, training=True):
        """
        Parameters
        ----------
        inputs : (batch_size, data_length)
        mixup_lambda : (batch_size,)
        """
        x = self.spectrogram_extractor(inputs) # (batch_size, time_steps, freq_bins)
        x = self.logmel_extractor(x) # (batch_size, time_steps, mel_bins)
        x = tf.expand_dims(x, axis = 1) # Second dimension is the number of channels

        # NOTE investigate or ask Qiuqiang. 
        x = tf.transpose(x, perm=[0, 3, 2, 1])
        x = self.bn0(x)
        x = tf.transpose(x, perm=[0, 3, 2, 1])

        # if training:
        #     x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if training:
            x = do_mixup(x,  tf.reshape(mixup_lambda,
                (mixup_lambda.shape[0], 1, 1, 1)))

        x = tf.transpose(x, perm=[0,3,2,1]) # channels at the end and inverting time-freq

        return x