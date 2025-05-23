"""
main script

"""

from Training import train

# number of epochs
EPOCHS = 50#55000

# initial learning rate
LR = 3e-4

# data_dir: the directory in which datasets are stored
data_dir = '../../Files/'

# name of dataset to be used
model_names = ['LSTM', 'RNN', 'GRU', 'CNN']

datasets = ['OSCMonoSquare', 'OSCMonoTri', 'OSCMonoSaw']

input_dims = [96]
units = [16]
BATCH_SIZEs = [512]
model_internal_dims = [4]

for model_name in model_names:
    for model_internal_dim in model_internal_dims:
        for BATCH_SIZE in BATCH_SIZEs:
            for input_dim in input_dims:
                  for dataset in datasets:
                        for unit in units:
                              train(data_dir=data_dir,
                                    save_folder=model_name+dataset + '_' + str(unit) + '_' + str(input_dim) + '_' + str(BATCH_SIZE) + '_' + str(model_internal_dim),
                                    dataset=dataset,
                                    batch_size=BATCH_SIZE,
                                    learning_rate=LR,
                                    input_dim=input_dim,
                                    units=unit,
                                    model_internal_dim=model_internal_dim,
                                    epochs=EPOCHS,
                                    model_name=model_name,
                                    inference=False)
