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

import librosa
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K

def FFT(y_true, y_pred):
    
    Y_true = tf.signal.fft(y_true)
    Y_pred = tf.signal.fft(y_pred)
    loss = tf.divide(tf.norm((Y_true - Y_pred), ord=1), tf.norm(Y_true, ord=1) + 1e-6)
    return loss
