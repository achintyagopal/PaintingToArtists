import os
import argparse
import sys
import pickle

from cs475_types import *

def get_args():

    parser = argparse.ArgumentParser(description="This is the main file for running different types of algorithms.")

    parser.add_argument("--folder", type=str,
        help="The folder to use for extracting features.")
    parser.add_argument("--data", type=str, 
        help="The file for training or testing.")
    parser.add_argument("--mode", type=str, required=True, choices=["feature", "train", "test"],
                        help="Operating mode: feature extraction, train or test.")
    parser.add_argument("--training-file", type=str,
                        help="The name of the file containing training data")
    parser.add_argument("--testing-file", type=str,
                        help="The name of the file containing test data")
    parser.add_argument("--model-file", type=str,
                        help="The name of the model file to create/load.")
    parser.add_argument("--predictions-file", type=str, help="The predictions file to create.")
    parser.add_argument("--feature_algorithm", type=str, help="The name of the algorithm for training.")
    parser.add_argument("--learning_algorithm", type=str, help="The name of the algorithm for training.")

    args = parser.parse_args()
    check_args(args)

    return args


def check_args(args):
    if args.mode.lower() == "feature":
        if args.folder is None:
            raise Exception("--folder (folder with data) should be specified in mode \"feature\"")
        if args.algorithm is None:
            raise Exception("--algorithm (feature extraction algorithm) should be specified in mode \"feature\"")
        if args.training_file is None:
            raise Exception("--training-file (save training features) should be specified in mode \"feature\"")
        if args.testing_file is None:
            raise Exception("--testing-file (save testing features) should be specified in mode \"feature\"")
        if args.feature_algorithm is None:
            raise Exception("--feature-algorithm should be specified in mode \"feature\"")
        if args.model_file is None:
            raise Exception("--model-file should be specified in mode \"feature\"")
    elif args.mode.lower() == "train":
        if args.folder is None and args.data is None:
            raise Exception("--data or --folder (load training features) should be specified in mode \"train\"")
        if args.folder is not None and args.feature_algorithm is None:
            raise Exception("--feature-algorithm should be specified in mode \"train\" when given folder")
        if args.training_file is None:
            raise Exception("--training-file (load training features) should be specified in mode \"train\"")
        if not os.path.exists(args.training_file):
            raise Exception("training file specified by --training-file does not exist.")
        if args.training_algorithm is None:
            raise Exception("--training-algorithm should be specified in mode \"train\"")
    else:
        if args.folder is None and args.data is None:
            raise Exception("--data or --folder (load training features) should be specified in mode \"test\"")
        if args.folder is not None and args.feature_algorithm is None:
            raise Exception("--feature-algorithm should be specified in mode \"test\" when given folder")
        if args.predictions_file is None:
            raise Exception("--prediction-file should be specified in mode \"test\"")
        if args.testing_file is None:
            raise Exception("--testing-file (load testing features) should be specified in mode \"test\"")
        if not os.path.exists(args.testing_file):
            raise Exception("testing file specified by --testing-file does not exist.")
        if args.model_file is None:
            raise Exception("--model-file should be specified in mode \"test\"")
        if not os.path.exists(args.model_file):
            raise Exception("model file specified by --model-file does not exist.")


# traverse through folder, extract features and convert to instances
def create_instances(folder_name, algorithm, args):
    # folders = check_directory_structure(folder_name)            
    foldersTmp = os.listdir(folder_name)
    folders = []
    for folder in foldersTmp:
        if folder[0] == '.':
            continue
        folders.append(folder)

    instances = []
    for folder in folders:
        path = train_path + folder + '/'
        files = os.listdir(path)
        for file in files:
            if os.path.isfile(os.path.join(path, file)):
                instnaces.append(create_instance(join(path, file), algorithm, args))

    return instances


# load instances from filename
def load_data(filename):
    instances = None
    try:
        with open(file, 'rb') as reader:
            instances = pickle.load(reader)
    except IOError:
        raise Exception("Exception while reading the model file.")
    except pickle.PickleError:
        raise Exception("Exception while loading pickle.")
    return instances


# print in predictions_file both correct answer and wrong answer
def write_predictions(predictor, instances, predictions_file):
    try:
        with open(predictions_file, 'w') as writer:
            for instance in instances:
                label = predictor.predict(instance)
        
                writer.write(str(label))
                writer.write(' ')
                writer.write(instance.get_label())
                writer.write('\n')
    except IOError:
        raise Exception("Exception while opening/writing file for writing predicted labels: " + predictions_file)


def create_instance(filename, algorithm, args):
    if algorithm == "color":
        # create color histogram
        return None
    elif algorithm == "raster":
        # just make one dimensional array
        return None
    elif algorithm == "bow":
        # create bag of words encoding
        return None
    return None


# train algorithm on instances
def train(instances, algorithm, args):
    if algorithm == "SVM":
        # create multiclass SVM model
        return None
    elif algorithm == "NeuralNetwork":
        # train a neural network
        return None
    return None


def main():
    args = get_args()

    mode = args.mode.lower()
    if mode == "feature":

        # args.folder is the folder where training data and testing data is, with specific
        # directory structure
        training_instances = create_instances(args.folder + '/train/', args.feature_algorithm, args)
        testing_instances = create_instances(args.folder + '/test/', args.feature_algorithm, args)

        # save features extracted for training
        try:
            with open(args.training_file, 'wb') as writer:
                pickle.dump(training_instances, writer)
        except IOError:
            raise Exception("Exception while writing to the model file.")        
        except pickle.PickleError:
            raise Exception("Exception while dumping pickle.")

        # save features extracted for testing
        try:
            with open(args.testing_file, 'wb') as writer:
                pickle.dump(testing_instances, writer)
        except IOError:
            raise Exception("Exception while writing to the model file.")        
        except pickle.PickleError:
            raise Exception("Exception while dumping pickle.")

    elif mode == "train":
        # Load the training data.
        if args.data is not None:
            instances = load_data(args.data)
        else:
            instances = create_instances(args.folder + '/train/', args.feature_algorithm, args)

        # train some model
        predictor = train(features, args.training_algorithm, args)

        # save the model
        try:
            with open(args.model_file, 'wb') as writer:
                pickle.dump(predictor, writer)
        except IOError:
            raise Exception("Exception while writing to the model file.")        
        except pickle.PickleError:
            raise Exception("Exception while dumping pickle.")
            
    elif mode == "test":
        # Load the test data.
        if args.data is not None:
            instances = load_data(args.data)
        else:
            instances = create_instances(args.folder + '/test/', args.algorithm, args)

        predictor = None
        # Load the model.
        try:
            with open(args.model_file, 'rb') as reader:
                predictor = pickle.load(reader)
        except IOError:
            raise Exception("Exception while reading the model file.")
        except pickle.PickleError:
            raise Exception("Exception while loading pickle.")
            
        # use predictor on instances, save in args.predictions_file
        write_predictions(predictor, instances, args.predictions_file)
    else:
        raise Exception("Unrecognized mode.")

if __name__ == "__main__":
    main()

