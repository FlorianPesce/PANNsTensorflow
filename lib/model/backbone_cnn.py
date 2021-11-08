import tensorflow as tf

class ConvBlock(tf.keras.layers.Layer):
    """
    Convolutional Block Class.
    """
    def __init__(self, out_channels, pool_size=(2, 2), pool_type='avg'):
        """
        Parameters
        ----------
        out_channels : int
            Number of output channels
        pool_size : int or tuple/list of 2 ints
            Size of the pooling window
        pool_type : string
            Type of pooling
        """
        super(ConvBlock, self).__init__()

        self.out_channels = out_channels
        
        self.conv1 = tf.keras.layers.Conv2D(filters=out_channels,
                                            kernel_size=3, strides=1,
                                            padding='same',
                                            use_bias=False,
                                            kernel_initializer='glorot_uniform')
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(filters=out_channels,
                                            kernel_size=3, strides=1,
                                            padding='same',
                                            use_bias=False,
                                            kernel_initializer='glorot_uniform')
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.pool_size = pool_size
        self.pool_type = pool_type
        if pool_type == 'max':
            self.pooling = tf.keras.layers.MaxPool2D(pool_size = pool_size)
        elif pool_type == 'avg':
            self.pooling = tf.keras.layers.AveragePooling2D(pool_size = pool_size)
        elif pool_type == 'avg+max':
            self.pooling1 = tf.keras.layers.AveragePooling2D(pool_size = pool_size)
            self.pooling2 = tf.keras.layers.MaxPool2D(pool_size = pool_size)
        else:
            raise ValueError(f"pool_type should be one of the following:"
            "max, avg or avg+max. Here, we got {pool_type}.")
        

    def call(self, inputs):
        """
        Parameters
        ----------
        inputs : (batch_size, height, width, channels)
        """

        x = inputs
        x = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        x = tf.keras.activations.relu(self.bn2(self.conv2(x)))

        if self.pool_type == 'avg+max':
            x1 = self.pooling1(x)
            x2 = self.pooling2(x)
            x = x1 + x2
        else:
            x = self.pooling(x)
        
        return x

class Cnn14(tf.keras.layers.Layer):
    """
    CNN14 Backbone
    """
    def __init__(self, dropout_rate=0.2, name="cnn14_backbone"):
        """
        Parameters
        ----------
        classes_num : int
            number of output classes
        dropout_rate : float, optional
            dropout used in backbone, by default 0.2
        include_top : bool, optional
            whether to include the clipwise_output or not, by default True
        """
        super(Cnn14, self).__init__()

        self.dropout_rate = dropout_rate

        self.conv_block2 = ConvBlock(out_channels=128)
        self.conv_block3 = ConvBlock(out_channels=256)
        self.conv_block4 = ConvBlock(out_channels=512)
        self.conv_block5 = ConvBlock(out_channels=1024)
        self.conv_block6 = ConvBlock(out_channels=2048)

        self.fc1 = tf.keras.layers.Dense(2048, use_bias=True,
            kernel_initializer='glorot_uniform')


    def call(self, inputs, training=True):
        """
        Parameters
        ----------
        inputs : (batch_size, height, width, channels)
        """

        x = inputs
        x = tf.keras.layers.Dropout(self.dropout_rate)(x, training=training) 
        x = self.conv_block2(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x, training=training) 
        x = self.conv_block3(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x, training=training) 
        x = self.conv_block4(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x, training=training) 
        x = self.conv_block5(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x, training=training) 
        x = self.conv_block6(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x, training=training) 
        x = tf.math.reduce_mean(x, axis=1) # freq dimension

        x1 = tf.math.reduce_max(x, axis=1) # time dimension
        x2 = tf.math.reduce_mean(x, axis=1) # time dimension
        x = x1 + x2
        x = tf.keras.layers.Dropout(.5)(x, training=training)
        x = tf.keras.activations.relu(self.fc1(x))
        embedding = tf.keras.layers.Dropout(.5)(x, training=training)
    
        return embedding



