from Training import train
import argparse


"""
main script

"""
def parse_args():
    parser = argparse.ArgumentParser(description='Trains an State-based network. Can also be used to run pure inference.')

    parser.add_argument('--model_save_dir', default='./models', type=str, nargs='?', help='Folder directory in which to store the trained models.')

    parser.add_argument('--data_dir', default='./datasets', type=str, nargs='?', help='Folder directory in which the datasets are stored.')

    parser.add_argument('--datasets', default=" ", type=str, nargs='+', help='The names of the datasets to use. [OSCMonoSquare, OSCMonoTri, OSCMonoSaw]')

    parser.add_argument('--model', default=" ", type=str, nargs='+', help='The name of the model to train (TCN, RNN, GRU, LSTM).')

    parser.add_argument('--epochs', default=60, type=int, nargs='?', help='Number of training epochs.')

    parser.add_argument('--batch_size', default=8, type=int, nargs='?', help='Batch size.')

    parser.add_argument('--input_size', default=96, type=int, nargs='?', help='The number of samples to use as input.')

    parser.add_argument('--model_internal_dim', default=4, type=int, nargs='?', help='The number of samples to be output of the compression layer (only apply to RNN, GRU and LSTM model).')

    parser.add_argument('--kernel_size', default=3, type=int, nargs='?', help='The size of the kernel in the TCN layers (only apply to TCN model).')

    parser.add_argument('--units', default=64, nargs='+', type=int, help='Hidden layer size (amount of units) of the network.')

    parser.add_argument('--learning_rate', default=3e-4, type=float, nargs='?', help='Initial learning rate.')

    parser.add_argument('--only_inference', default=False, type=bool, nargs='?', help='When True, skips training and runs only inference on the pre-model. When False, runs training and inference on the trained model.')

    return parser.parse_args()


def start_train(args):

    print("######### Preparing for training/inference #########")
    print("\n")
    train(data_dir=args.data_dir,
          model_save_dir=args.model_save_dir,
          save_folder=f'ED_{args.dataset}_{args.units}',
          dataset=args.datasets,
          model=args.model,
          epochs=args.epochs,
          batch_size=args.batch_size,
          input_dim=args.input_size,
          model_internal_dim=args.model_internal_dim,
          kernel_size=args.kernel_size,
          units=args.units,
          learning_rate=args.learning_rate,
          inference=args.only_inference)


def main():
    args = parse_args()
    start_train(args)

if __name__ == '__main__':
    main()