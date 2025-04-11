import librosa
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K

def FFT(y_true, y_pred):
    
    Y_true = tf.signal.fft(y_true)
    Y_pred = tf.signal.fft(y_pred)
    loss = tf.divide(tf.norm((Y_true - Y_pred), ord=1), tf.norm(Y_true, ord=1) + 1e-6)
    return loss
