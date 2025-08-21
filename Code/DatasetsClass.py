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
# R. Simionato, 2025, "Towards Neural Emulation of Voltage-Controlled Oscillator" in proceedings of the 25th Digital Audio Effect Conference, Ancona, Italy.

import pickle
import os
import numpy as np
from tensorflow.keras.utils import Sequence
import tensorflow as tf


class DataGeneratorPickles(Sequence):

    def __init__(self, data_dir, filename, input_size, model, set='train', batch_size=10):
        """
        Initializes a data generator object
          :param data_dir: the directory in which data are stored
          :param filename: the name of the dataset
          :param input_size: the input size
          :param cond: the number of conditioning values
          :param batch_size: The size of each batch returned by __getitem__
        """

        self.data_dir = data_dir
        self.filename = filename
        self.batch_size = batch_size
        self.input_size = input_size
        self.model = model
        # prepare the input, target and conditioning matrix
        self.ys, self.zs = self.prepareXYZ(data_dir, filename)

        self.training_steps = (self.ys.shape[1] // self.batch_size) * self.ys.shape[0]
        self.lim = (self.ys.shape[1] // self.batch_size) * self.batch_size

        self.x = tf.signal.frame(self.ys[0, :self.lim], input_size, 1).numpy()
        self.y = tf.signal.frame(self.ys[0, input_size:self.lim + input_size], 1, 1).numpy()[:self.x.shape[0]]
        self.z = tf.signal.frame(self.zs[0, input_size:self.lim + input_size], 1, 1).numpy()

        for i in range(1, self.ys.shape[0]):
            x_temp = tf.signal.frame(self.ys[i, :self.lim], input_size, 1).numpy()
            self.x = np.concatenate((self.x, x_temp), axis=0)
            y_temp = tf.signal.frame(self.ys[i, input_size:self.lim + input_size], 1, 1).numpy()[:x_temp.shape[0]]
            self.y = np.concatenate((self.y, y_temp), axis=0)
            z_temp = tf.signal.frame(self.zs[i, input_size:self.lim + input_size], 1, 1).numpy()
            self.z = np.concatenate((self.z, z_temp), axis=0)

        del self.ys, self.zs

        if set == 'train':
            noise = tf.random.normal([self.x.shape[0], self.x.shape[1]], mean=0., stddev=1.0, seed=422,
                                     dtype=tf.float32, name='noise')
            noise = noise / tf.math.reduce_max(noise)
            noise = noise.numpy()
            noise = noise * 0.1
            self.x = noise + self.x

        self.maxl = y_temp.shape[0]
        self.on_epoch_end()

    def prepareXYZ(self, data_dir, filename):

        # load all the audio files
        file_data = open(os.path.normpath('/'.join([data_dir, filename])), 'rb')
        Z = pickle.load(file_data)
        y = np.array(Z['y'][:], dtype=np.float32)
        z = np.array(Z['z'][:], dtype=np.float32)
        y = y / np.max(np.abs(y))
        return y, z

    def on_epoch_end(self):
        # create/reset the vector containing the indices of the batches
        self.indices = np.arange(0, self.y.shape[0])

    def __len__(self):
        # compute the needed number of iterations before conclude one epoch
        return int(self.y.shape[0] / self.batch_size) - 1

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.on_epoch_end()

    def __getitem__(self, idx):

        # get the indices of the requested batch
        indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        # fill the batches
        X = self.x[indices].reshape(self.batch_size, self.input_size, 1)
        Y = self.y[indices].reshape(self.batch_size, 1)
        Z = self.z[indices].reshape(self.batch_size, 1)

        return [Z, X], Y