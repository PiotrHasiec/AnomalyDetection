import tensorflow as tf
# import sklearn
import pandas as pd
import numpy as np
class Autoencoder(tf.keras.Model):
  def __init__(self, inputs_shape,laten_dim,noise):
    super(Autoencoder, self).__init__()
    self.inputs_shape = (inputs_shape[0],inputs_shape[1])  
    self.model = tf.keras.Sequential()
    self.model.add(tf.keras.layers.InputLayer(self.inputs_shape))
    self.model.add(tf.keras.layers.Flatten())
    self.model.add(tf.keras.layers.GaussianNoise(noise))
    # self.model.add(tf.keras.layers.Dense(2*laten_dim*inputs_shape[0],activation='relu',activity_regularizer=tf.keras.regularizers.L2(1e-5)))
    # self.model.add(tf.keras.layers.BatchNormalization())
    self.model.add(tf.keras.layers.Dense(int(laten_dim*inputs_shape[0]),activation='relu',activity_regularizer=tf.keras.regularizers.L1(1e-5)))
    # self.model.add(tf.keras.layers.BatchNormalization())
    # self.model.add(tf.keras.layers.Dense(2*laten_dim*inputs_shape[0],activation='relu',activity_regularizer=tf.keras.regularizers.L2(1e-5)))
    self.model.add(tf.keras.layers.Dense(inputs_shape[0]*inputs_shape[1]))
    self.model.add(tf.keras.layers.Reshape(self.inputs_shape))

  def call(self, x):

    decoded = self.model(x)
    return decoded


class MultiStepLSTM(tf.keras.Model):
  def __init__(self,seq_len_out,latten_dim,num_features):
    super(MultiStepLSTM, self).__init__()
    self.model = tf.keras.Sequential([
    tf.keras.layers.LSTM(latten_dim, return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(int(latten_dim),activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(seq_len_out*num_features,
                          kernel_initializer=tf.initializers.zeros(),activation='linear'),
    # Shape => [batch, out_steps, features].
    tf.keras.layers.Reshape([seq_len_out,num_features])
])


class SingleStepLSTM(tf.keras.Model):
  def __init__(self,latten_dim,num_features,return_sequences):
    super(SingleStepLSTM, self).__init__()
    self.model = tf.keras.Sequential([
    tf.keras.layers.LSTM(latten_dim, return_sequences=return_sequences),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(int(0.5*num_features),activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_features,activation='linear'),
])

class LSTMAutoencoder(tf.keras.Model):
  def __init__(self, n_timestep,n_features,nuber_of_units):
    super(LSTMAutoencoder, self).__init__()
    self.model = tf.keras.Sequential()
    self.model.add(tf.keras.layers.GaussianNoise(0.5))
    self.model.add(tf.keras.layers.Dropout(0.25))
    self.model.add( tf.keras.layers.LSTM(nuber_of_units, return_sequences=True,activity_regularizer=tf.keras.regularizers.L2(1e-5)))
    self.model.add(tf.keras.layers.TimeDistributed( tf.keras.layers.Dense(n_features,activation='linear')))
    self.model.add( tf.keras.layers.LSTM(n_features, return_sequences=True,activity_regularizer=tf.keras.regularizers.L2(1e-5)))
    self.model.add(tf.keras.layers.TimeDistributed( tf.keras.layers.Dense(n_features,activation='linear')))
    # self.model.add(tf.keras.layers.Reshape([-1,n_timestep,n_features]))
  def call(self, x):
    
    return self.model(x)
    
