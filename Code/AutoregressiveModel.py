# Copyright (C) 2025 Riccardo Simionato, University of Oslo
# Inquiries: riccardo.simionato.vib@gmail.com.com
#
# This code is free software: you can redistribute it and/or modify it under the terms
# of the GNU Lesser General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Less General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along with this code.
# If not, see <http://www.gnu.org/licenses/>.
#
# If you use this code or any part of it in any program or publication, please acknowledge
# its authors by adding a reference to this publication:
#
# R. Simionato, 2025, "Towards Neural Emulation of Voltage-Controlled Oscillator" in proceedings of the 23th Digital Audio Effect Conference, Ancona, Italy.

import tensorflow as tf
from Layers import FiLM


class LSTM_OSC(tf.keras.layers.Layer):
    def __init__(self, units, input_size, batch_size, model_internal_dim=16, type=tf.float32):
        super(LSTM_OSC, self).__init__()

        self.type = type
        self.batch_size = batch_size
        self.input_size = input_size
        self.units = units
        
        self.lin = tf.keras.layers.Dense(units=model_internal_dim, name='lin_layer')
        self.rec = tf.keras.layers.LSTM(units=units, return_sequences=False, name='RNN')
        self.out = tf.keras.layers.Dense(units=1, activation='tanh', name='output_layer')
        self.h = tf.keras.layers.Dense(units=units, activation='tanh')
        self.c = tf.keras.layers.Dense(units=units, activation='tanh')
        self.film = FiLM(units)

    def call(self, input):
    
        cond = input[1]
        h = self.h(cond)
        c = self.c(cond)

        inp = self.lin(input[0][:, :, 0])
        inp = tf.expand_dims(inp, axis=-1)
        out = self.rec(inp, initial_state=[h, c])
        out = self.film(out, input[1])
        out = self.out(out)

        return out
        
class GRU_OSC(tf.keras.layers.Layer):
    def __init__(self, units, input_size, batch_size, model_internal_dim, type=tf.float32):
        super(GRU_OSC, self).__init__()

        self.type = type
        self.batch_size = batch_size
        self.input_size = input_size
        self.units = units
        self.lin = tf.keras.layers.Dense(units=model_internal_dim, name='lin_layer')
        self.rec = tf.keras.layers.GRU(units=units, return_state=False, return_sequences=False, name='OSC')
        self.out = tf.keras.layers.Dense(units=1, activation='tanh', name='output_layer')
        self.h = tf.keras.layers.Dense(units=units, activation='tanh')
        self.film = FiLM(units)

    def call(self, input):
        cond = input[1]
        h = self.h(cond)

        inp = self.lin(input[0][:,:,0])
        inp = tf.expand_dims(inp, axis=-1)
        out = self.rec(inp, initial_state=h)
        
        out = self.film(out, cond)
        out = self.out(out)
        return out

class RNN_OSC(tf.keras.layers.Layer):
    def __init__(self, units, input_size, batch_size, model_internal_dim, type=tf.float32):
        super(RNN_OSC, self).__init__()

        self.type = type
        self.batch_size = batch_size
        self.input_size = input_size
        self.units = units

        self.lin = tf.keras.layers.Dense(units=model_internal_dim, name='lin_layer')
        self.rec = tf.keras.layers.SimpleRNN(units=self.units, return_state=False, return_sequences=False, activation=None, name='OSC')
        self.out = tf.keras.layers.Dense(units=1, activation='tanh', name='output_layer')
        self.h = tf.keras.layers.Dense(units=self.units, activation='tanh')
        self.film = FiLM(units)

    def call(self, input):
        cond = input[1]
        h = self.h(cond)

        inp = self.lin(input[0][:,:,0])
        inp = tf.expand_dims(inp, axis=-1)
        out = self.rec(inp, initial_state=h)
        
        out = self.film(out, cond)
        out = self.out(out)

        return out

class TCN_OSC(tf.keras.layers.Layer):
    def __init__(self, units, input_size, kernel_size, batch_size, type=tf.float32):
        super(TCN_OSC, self).__init__()

        self.type = type
        self.batch_size = batch_size
        self.input_size = input_size
        self.units = units

        # Create TCN residual blocks with increasing dilation
        # Causal dilated convolution
        self.cnn1 = tf.keras.layers.Conv1D(filters=units,
                   kernel_size=kernel_size,
                   dilation_rate=2,
                   padding='causal')

        # Second dilated causal convolution
        self.cnn2 = tf.keras.layers.Conv1D(filters=units,
                   kernel_size=kernel_size,
                   dilation_rate=4,
                   padding='causal')

        # Third dilated causal convolution
        self.cnn3 = tf.keras.layers.Conv1D(filters=units,
                   kernel_size=kernel_size,
                   dilation_rate=8,
                   padding='causal')

        # Forth dilated causal convolution
        self.cnn4 = tf.keras.layers.Conv1D(filters=units,
                    kernel_size=kernel_size,
                    dilation_rate=16,
                    padding='causal')

        self.take_last = tf.keras.layers.Lambda(lambda z: z[:, -1, :])
        self.cnn1d_1 =  tf.keras.layers.Conv1D(units, kernel_size=1, padding='same')
        self.cnn1d_2 =  tf.keras.layers.Conv1D(units, kernel_size=1, padding='same')
        self.cnn1d_3 =  tf.keras.layers.Conv1D(units, kernel_size=1, padding='same')
        self.cnn1d_4 =  tf.keras.layers.Conv1D(units, kernel_size=1, padding='same')

        self.out = tf.keras.layers.Dense(units=1, activation='tanh', name='output_layer')
        self.film = FiLM(self.units)

    def call(self, input):
        # First dilated causal convolution
        prev_x = input[0]
        cond = input[1]

        # Causal dilated convolution
        out =  self.cnn1(prev_x)
        out =  tf.keras.layers.Activation('relu')(out)
        prev_x = self.cnn1d_1(prev_x)
        out = tf.keras.layers.add([prev_x, out])
        prev_x = out

        out =  self.cnn2(out)
        out =  tf.keras.layers.Activation('relu')(out)
        prev_x = self.cnn1d_2(prev_x)
        out = tf.keras.layers.add([prev_x, out])
        prev_x = out

        out =  self.cnn3(out)
        out =  tf.keras.layers.Activation('relu')(out)
        prev_x = self.cnn1d_3(prev_x)
        out = tf.keras.layers.add([prev_x, out])
        prev_x = out

        out =  self.cnn4(out)
        out =  tf.keras.layers.Activation('relu')(out)
        prev_x = self.cnn1d_4(prev_x)
        out = tf.keras.layers.add([prev_x, out])

        out = self.take_last(out)
        out = self.film(out, cond)
        out = self.out(out)

        return out
        
def create_model(units, input_size, kernel_size, batch_size, mode, model_internal_dim):

    # Defining inputs
    inputs = tf.keras.layers.Input(batch_shape=(batch_size, input_size, 1), name='input')
    conds = tf.keras.layers.Input(batch_shape=(batch_size, 1), name='conds')

    if mode == 'LSTM':
        outputs = LSTM_OSC(units=units, input_size=input_size, batch_size=batch_size, model_internal_dim=model_internal_dim)([inputs, conds])
    elif mode == 'RNN':
        outputs = RNN_OSC(units=units, input_size=input_size, batch_size=batch_size, model_internal_dim=model_internal_dim)([inputs, conds])
    elif mode == 'GRU':
        outputs = GRU_OSC(units=units, input_size=input_size, batch_size=batch_size, model_internal_dim=model_internal_dim)([inputs, conds])
    elif mode == 'TCN':
        outputs = TCN_OSC(units=units,  input_size=input_size, batch_size=batch_size, kernel_size=kernel_size)([inputs, conds])
               
    model = tf.keras.models.Model([conds, inputs], outputs)

    model.summary()

    return model
