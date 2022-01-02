''' 
Model builder function and associated AR layers.

Extracts features from each of the three images, creating three feature vectors; x1, x2, x3.

Fits an autoregressive model to those three feature vectors to predict the future feature vector: 
	x_{4} = b_{0} + b_{1} x_{1} + b_{2} f_{2} + b_{3} x_{3}

The model takes as its input:
	- 3 images
	- 3 time intervals from the image observation time to the prediction time

joshua.bridge@liverpool.ac.uk
'''
from keras import layers, models, backend, applications, regularizers, initializers
from keras.layers import Layer    
import tensorflow as tf


class coeff(Layer):

    def __init__(self, **kwargs):
        super(coeff, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(2048,2048,),
                                      initializer=initializers.he_normal(),
                                      trainable=True)
        super(coeff, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return tf.linalg.matvec(self.kernel, x)

    def compute_output_shape(self, input_shape):
        return input_shape  


class intercept(Layer):
    def __init__(self, **kwargs):
        super(intercept, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.bias = self.add_weight(shape=(2048,),
                                    initializer=initializers.he_normal(),
                                    name='bias')
        super(intercept, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
      output = x + self.bias
      return output

    def compute_output_shape(self, input_shape):
        return input_shape    


def build():
	inputs = layers.Input(shape=(3, 256, 256, 3))
	time3 = layers.Input(shape=(1,))
	time2 = layers.Input(shape=(1,))
	time1 = layers.Input(shape=(1,))

	x = layers.TimeDistributed(applications.InceptionV3(include_top=False, weights='imagenet', pooling=None))(inputs)
	x = layers.Dropout(0.5)(x)
	x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)

	


	x3 = layers.Lambda(lambda x: x[:, 0, :])(x)
	x3 = layers.Reshape((2048,))(x3)
	x3 = coeff()(x3)
	x3 = layers.Multiply()([time3, x3])
	x2 = layers.Lambda(lambda x: x[:, 1, :])(x)
	x2 = layers.Reshape((2048,))(x2)
	x2 = coeff()(x2)
	x2 = layers.Multiply()([time2, x2])
	x1 = layers.Lambda(lambda x: x[:, 2, :])(x)
	x1 = layers.Reshape((2048,))(x1)
	x1 = coeff()(x1)
	x1 = layers.Multiply()([time1, x1])
	mod = layers.Add()([x1, x2, x3])

	mod = intercept()(mod)

	out = layers.Dense(1, activation='sigmoid', name='label',
	                            kernel_regularizer=regularizers.L1L2(l1=0, l2=2e-5))(mod)


  	
	model = models.Model([inputs, time3, time2, time1], out)
			

	return model 


