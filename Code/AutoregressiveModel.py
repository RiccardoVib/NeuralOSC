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


class CNN_OSC(tf.keras.layers.Layer):
    def __init__(self, units, input_size, batch_size, type=tf.float32):
        super(CNN_OSC, self).__init__()

        self.type = type
        self.batch_size = batch_size
        self.input_size = input_size
        self.units = units
        self.rec = tf.keras.layers.Conv1D(filters=self.units, kernel_size=input_size, name='OSC')
        self.out = tf.keras.layers.Dense(units=1, activation='tanh', name='output_layer')
        self.film = FiLM(self.units)

    def call(self, input):
        cond = input[1]

        out = self.rec(input[0])

        out = self.film(out[:,0,:], cond)
        out = self.out(out)

        return out

        
def create_model(units, input_size, batch_size, mode, model_internal_dim):

    # Defining inputs
    inputs = tf.keras.layers.Input(batch_shape=(batch_size, input_size, 1), name='input')
    conds = tf.keras.layers.Input(batch_shape=(batch_size, 1), name='conds')

    if mode == 'LSTM':
        outputs = LSTM_OSC(units, input_size, batch_size, model_internal_dim)([inputs, conds])
    elif mode == 'RNN':
        outputs = RNN_OSC(units + 28, input_size, batch_size, model_internal_dim)([inputs, conds])
    elif mode == 'GRU':
        outputs = GRU_OSC(units+units//8, input_size, batch_size, model_internal_dim)([inputs, conds])
    elif mode == 'CNN':
        outputs = CNN_OSC(units*8, input_size, batch_size)([inputs, conds])
               
    model = tf.keras.models.Model([conds, inputs], outputs)

    model.summary()

    return model
