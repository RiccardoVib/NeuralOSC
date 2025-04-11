import os
import tensorflow as tf
from UtilsForTrainings import plotTraining, writeResults, checkpoints, predictWaves, MyLRScheduler
from AutoregressiveModel import create_model
from DatasetsClass import DataGeneratorPickles
import numpy as np
from Metrics import FFT
from LossFunctions import NMSELoss
import sys
import time
import matplotlib.pyplot as plt


def train(**kwargs):
    """
      :param data_dir: the directory in which dataset are stored [string]
      :param save_folder: the directory in which the models are saved [string]
      :param batch_size: the size of each batch [int]
      :param learning_rate: the initial leanring rate [float]
      :param units: the number of model's units [int]
      :param input_size: the input size [int]
      :param model_save_dir: the directory in which models are stored [string]
      :param save_folder: the directory in which the model will be saved [string]
      :param inference: if True it skip the training and it compute only the inference [bool]
      :param dataset: name of the datset to use [string]
      :param epochs: the number of epochs [int]
      :param fs: the sampling rate [int]
    """

    batch_size = kwargs.get('batch_size', 1)
    learning_rate = kwargs.get('learning_rate', 1e-1)
    units = kwargs.get('units', 16)
    input_size = kwargs.get('input_dim', 1)
    model_save_dir = kwargs.get('model_save_dir', '../../TrainedModels')
    save_folder = kwargs.get('save_folder', 'ED_Testing')
    model_name = kwargs.get('model_name', '')
    inference = kwargs.get('inference', False)
    dataset = kwargs.get('dataset', None)
    data_dir = kwargs.get('data_dir', '../../../Files/')
    epochs = kwargs.get('epochs', 60)
    fs = kwargs.get('fs', 48000)
    model_internal_dim = kwargs.get('model_internal_dim', 4)

    # set all the seed in case reproducibility is desired
    #np.random.seed(42)
    #tf.random.set_seed(42)
    #random.seed(42)

    # start the timer for all the training script
    # check if GPUs are available and set the memory growing
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpu = tf.config.experimental.list_physical_devices('GPU')
    if len(gpu) != 0:
        tf.config.experimental.set_memory_growth(gpu[0], True)
    # tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=18000)])

    
    model = create_model(units=units, input_size=input_size, batch_size=batch_size, mode=model_name, model_internal_dim=model_internal_dim)


    # define callbacks: where to store the weights
    callbacks = []
    ckpt_callback, ckpt_callback_latest, ckpt_dir, ckpt_dir_latest = checkpoints(model_save_dir, save_folder)

    test_gen = DataGeneratorPickles(data_dir, dataset + '.pickle', input_size=input_size, model=model, set='test',
                                    batch_size=batch_size)
    # compile the model with the optimizer and selected loss function
    loss = NMSELoss()
        
    # if inference is True, it jump directly to the inference section without train the model
    if not inference:
        callbacks += [ckpt_callback, ckpt_callback_latest]#, earlystopping]
        # load the weights of the last epoch, if any
        last = tf.train.latest_checkpoint(ckpt_dir_latest)
        if last is not None:
            print("Restored weights from {}".format(ckpt_dir_latest))
            model.load_weights(last)
        else:
            # if no weights are found,the weights are random generated
            print("Initializing random weights.")


        train_gen = DataGeneratorPickles(data_dir, dataset + '.pickle', input_size=input_size, model=model,
                                         batch_size=batch_size)
        # the number of total training steps
        training_steps = train_gen.training_steps
        # define the Adam optimizer with initial learning rate, training steps
        opt = tf.keras.optimizers.Adam(
            learning_rate=MyLRScheduler(learning_rate, training_steps, epochs))

        model.compile(loss=loss, metrics=['mae', 'mse'], optimizer=opt)

        # training loop
        start = time.time()
        # defining the array taking the training and validation losses
        loss_training = np.zeros(epochs)
        loss_val = np.zeros(epochs)
        best_loss = 1e9
        # counting for early stopping
        count = 0

        for i in range(epochs):
            print('epochs', i)
            print(model.optimizer.learning_rate)

            results = model.fit(train_gen, epochs=1, verbose=0, shuffle=True, validation_data=test_gen,
                                callbacks=callbacks)
            # store the training and validation loss
            loss_training[i] = results.history['loss'][-1]
            loss_val[i] = results.history['val_loss'][-1]
            print(results.history['val_loss'][-1])

            # if validation loss is smaller then the best loss, the early stopping counting is reset
            if results.history['val_loss'][-1] < best_loss:
                best_loss = results.history['val_loss'][-1]
                count = 0
            # if not count is increased by one and if equal to 20 the training is stopped
            else:
                count = count + 1
                if count == 50:
                    break

        # store the training and validation loss
        loss_training = loss_training[:i]
        loss_val = loss_val[:i]

        avg_time_epoch = (time.time() - start)
        sys.stdout.write(f" Average time/epoch {'{:.3f}'.format(avg_time_epoch / 60)} min")
        sys.stdout.write("\n")

        # write and save results
        writeResults(results, units, epochs, batch_size, learning_rate, model_save_dir,
                     save_folder, epochs)

        # plot the training and validation loss for all the training
        plotTraining(loss_training[:i], loss_val[:i], model_save_dir, save_folder, str(epochs))

        print("Training done")

    # load the best weights of the model
    best = tf.train.latest_checkpoint(ckpt_dir)
    if best is not None:
        print("Restored weights from {}".format(ckpt_dir))
        model.load_weights(best).expect_partial()
    else:
        # if no weights are found,there is something wrong
        print("Something is wrong.")

    # real test
    model = create_model(units=units, input_size=input_size, batch_size=1, mode=model_name, model_internal_dim=model_internal_dim)

    # load the best weights of the model
    best = tf.train.latest_checkpoint(ckpt_dir)
    if best is not None:
        print("Restored weights from {}".format(ckpt_dir))
        model.load_weights(best).expect_partial()
    else:
        # if no weights are found,there is something wrong
        print("Something is wrong.")

    # predict the test set
    test_gen = DataGeneratorPickles(data_dir, dataset + '.pickle', input_size=input_size, model=model, set='test',
                                    batch_size=1)
    for j in range(0, test_gen.x.shape[0], test_gen.maxl):

        predictions = np.zeros(3000+input_size, dtype=np.float32)
        x = test_gen.x[j]
        predictions[:input_size] = x
        z = test_gen.z[j, 0].reshape(1, 1)

        prediction = model.predict([z, x.reshape(1, input_size, 1)], verbose=0)
        predictions[input_size] = prediction[0, -1]

        for i in range(1, 3000):
            p = predictions[i:i + input_size]
            prediction = model.predict([z, p.reshape(1, input_size, 1)], verbose=0)
            predictions[i + input_size] = prediction[0, -1]

        predictions = predictions[input_size:].reshape(-1)
        x = test_gen.x[j:j+len(predictions), -1].reshape(-1)
        y = test_gen.y[j:j+len(predictions), -1].reshape(-1)
        # plot and render the output audio file, together with the input and target
        predictWaves(predictions.reshape(1, -1), x.reshape(1, -1), y.reshape(1, -1), model_save_dir, save_folder, fs,
                     'autoregressive' + str(j//test_gen.maxl))
                     
                     
        # compute the metrics: mse, mae, esr and rmse
        nmse = tf.get_static_value(loss(y, predictions))
        fft = tf.get_static_value(FFT(y, predictions))

        # writhe and store the metrics values
        results_ = {'nmse': nmse, 'fft': fft}
        with open(os.path.normpath('/'.join([model_save_dir, save_folder, str(j//test_gen.maxl) + 'results.txt'])), 'w') as f:
            for key, value in results_.items():
                print('\n', key, '  : ', value, file=f)
        

    print('end')

    return 42
