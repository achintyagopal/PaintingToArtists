import os
import argparse
import sys
# import pickle
import cv2
import numpy as np
import dill as pickle
import math
import time
import multiprocessing
import ipyparallel

from project_types import *
from images import *
import ColorHistogram #import ColorHistogram
import BagOfWords #import BagOfWords
import HOG #import HOG
import MultiSVM #import MultiSVM
import structuredSVM #import StructuredSVM
import cnn #import CNN
import OriginalImg #import OriginalImg

def get_args():

    parser = argparse.ArgumentParser(description="This is the main file for running different types of algorithms.")

    parser.add_argument("--folder", type=str,
        help="The folder to use for extracting features.")
    parser.add_argument("--mode", type=str, required=True, choices=["feature", "train", "test"],
                        help="Operating mode: feature extraction, train or test.")
    parser.add_argument("--feature-file", type=str,
                        help="The name of the file containing feature converter/ instance creator")
    parser.add_argument("--model-file", type=str,
                        help="The name of the model file to create/load.")
    parser.add_argument("--predictions-file", type=str, help="The predictions file to create.")
    parser.add_argument("--feature-algorithm", type=str, 
                    help="The name of the algorithm for training.", 
                    choices=["color", "hog", "bow", "img"])
    parser.add_argument("--training-algorithm", type=str, 
                    help="The name of the algorithm for training.", 
                    choices=["mc_svm", "struct_svm", "quad_kernel", "rbf_kernel", "cnn"])
    parser.add_argument("--iterations", type=int,
                    help="The number of training iterations.", default=5)
    parser.add_argument("--bits", type=int,
                    help="The number of training iterations.", default=12)
    parser.add_argument("--clusters", type=int,
                    help="The number of training iterations.", default=800)
    
    parser.add_argument("--parallel", type=str,
                    help="Parallel or Serial run.", default="True")

    parser.add_argument("--procs", type=int,
                    help="The number of processes to use.", default=multiprocessing.cpu_count())

    parser.add_argument("--platform", type=str,
                    help="Parallel Platform to run on. Note: if ipython please make sure to run ipcluster before start of program.", 
                    default="native", choices=["native", "ipython"])

    parser.add_argument("--direct", type=str,
                    help="True to use ipython direct view, False to use load-balanced view.", 
                    default="True")

    args = parser.parse_args()
    check_args(args)

    return args


def check_args(args):
    if args.platform != "native" and args.platform != "ipython":
        raise Exception("--platform (native/ipython) not given supported platform")
    if args.mode.lower() == "feature":
        if args.folder is None:
            raise Exception("--folder (folder with data) should be specified in mode \"feature\"")
        if args.feature_algorithm is None:
            raise Exception("--algorithm (feature extraction algorithm) should be specified in mode \"feature\"")
        if args.feature_file is None:
            raise Exception("--feature-file (save features) should be specified in mode \"feature\"")
        if args.feature_algorithm is None:
            raise Exception("--feature-algorithm should be specified in mode \"feature\"")
    elif args.mode.lower() == "train":
        if args.folder is None and args.feature_file is None:
            raise Exception("--feature-file or --folder (load training features) should be specified in mode \"train\"")
        if args.folder is not None and args.feature_algorithm is None:
            raise Exception("--feature-algorithm should be specified in mode \"train\" when given folder")
        if args.feature_file is not None and not os.path.exists(args.feature_file):
            raise Exception("feature file specified by --feature-file does not exist.")
        if args.training_algorithm is None:
            raise Exception("--training-algorithm should be specified in mode \"train\"")
        if args.model_file is None:
            raise Exception("--model-file should be specified in mode \"train\"")
    else:
        if args.folder is None and args.feature_file is None:
            raise Exception("--feature-file or --folder (load training features) should be specified in mode \"train\"")
        if args.folder is not None and args.feature_algorithm is None:
            raise Exception("--feature-algorithm should be specified in mode \"test\" when given folder")
        if args.feature_file is not None and not os.path.exists(args.feature_file):
            raise Exception("feature file specified by --feature-file does not exist.")
        if args.predictions_file is None:
            raise Exception("--prediction-file should be specified in mode \"test\"")
        if args.model_file is None:
            raise Exception("--model-file should be specified in mode \"test\"")
        if not os.path.exists(args.model_file):
            raise Exception("model file specified by --model-file does not exist.")


def get_files(folder_name, algorithm, args):
    foldersTmp = os.listdir(folder_name)
    folders = []
    for folder in foldersTmp:
        if folder[0] == '.':
            continue
        folders.append(folder)

    imgs = []
    for folder in folders:
        path = folder_name + folder + '/'
        if not os.path.isdir(path):
            continue
            
        files = os.listdir(path)
        for file_str in files:
            complete_file_str = str((os.path.join(path, file_str)))
            if os.path.isfile(complete_file_str) and (complete_file_str.endswith('.jpg') or complete_file_str.endswith('.JPG')):
                imgs.append((os.path.join(path, file_str), folder))
    return imgs


# load instances from filename
def load_data(filename):
    instances = None
    try:
        with open(filename, 'rb') as reader:
            instances = pickle.load(reader)
    except IOError:
        raise Exception("Exception while reading the model file.")
    except pickle.PickleError:
        raise Exception("Exception while loading pickle.")
    return instances


# print in predictions_file both correct answer and wrong answer
def write_predictions(predictor, feature_converter, predictions_file):
    labels = predictor.predict(feature_converter)
    try:
        total = 0
        correct = 0
        with open(predictions_file, 'w') as writer:
            for i in range(len(labels)):
                label = labels[i]
                instance = feature_converter.getTestingInstance(i)
                
                writer.write(str(label))
                writer.write(' ')
                writer.write(instance.get_label())
                writer.write('\n')

                if str(label) == str(instance.get_label()):
                    correct += 1
                total +=1
    except IOError:
        raise Exception("Exception while opening/writing file for writing predicted labels: " + predictions_file)
    # print correct/float(total)

    # print "Testing Accuracy: ", (correct / float(total) * 100), "%"

def get_instance_converter(algorithm, args):
    if algorithm == "color":
        return ColorHistogram.ColorHistogram(args.bits)
    elif algorithm == "bow":
        return BagOfWords.BagOfWords(args.clusters)
    elif algorithm == "img":
        return OriginalImg.OriginalImg()
    elif algorithm == "hog":
        return HOG.HOG()
    return None

# train algorithm on instances
def train(algorithm, args):
    if algorithm == "mc_svm":
        # create multiclass SVM model
        return MultiSVM.MultiSVM()
    elif algorithm == "struct_svm":
        return structuredSVM.StructuredSVM(args.iterations)
    elif algorithm == "quad_kernel":
        return structuredSVM.StructuredSVM(lambda_fn = lambda x,y: x.dot(y) ** 2)
    elif algorithm == "rbf_kernel":
        return structuredSVM.StructuredSVM(lambda_fn = lambda x,y: math.e ** (-np.linalg.norm(x-y) ** 2/(2000)))
    elif algorithm == "cnn":
        # train a neural network
        nn = cnn.CNN((128,128,3))
        nn.add_convolution_layer(nodes = 32, size = (3,3))
        nn.add_relu_layer()
        nn.add_pool_layer(shape = (1,1,2))
        nn.add_convolution_layer(nodes = 1, size = (3,3))
        nn.add_fc_output_layer(nodes = 6)
        return nn
    return None


def main():
    args = get_args()

    mode = args.mode.lower()
    if mode == "feature":

        # args.folder is the folder where training data and testing data is, with specific
        # directory structure
        feature_converter = get_instance_converter(args.feature_algorithm, args)
        training_files = get_files(args.folder + '/train/', args.feature_algorithm, args)
        testing_files = get_files(args.folder + '/test/', args.feature_algorithm, args)

        # parallelize
        if args.parallel.lower() == "false":
            feature_converter.createTrainingInstances(training_files)
            feature_converter.createTestingInstances(testing_files)
        elif args.platform == "native":
            if args.feature_algorithm == "color":
                bits = feature_converter.bits
                train_inst = ColorHistogram.native_par_color(training_files, bits, args.procs)
                test_inst = ColorHistogram.native_par_color(testing_files, bits, args.procs)
                feature_converter.setTrainingInstances(train_inst)
                feature_converter.setTestingInstances(test_inst)
            elif args.feature_algorithm == "bow":
                # TODO
                BagOfWords.BagOfWords(args.clusters)
            elif args.feature_algorithm == "img":
                feature_converter.createTrainingInstances(training_files)
                feature_converter.createTestingInstances(testing_files)
            elif args.feature_algorithm == "hog":
                train_inst = HOG.native_par_hog(training_files, args.procs)
                test_inst = HOG.native_par_hog(testing_files, args.procs)
                feature_converter.setTrainingInstances(train_inst)
                feature_converter.setTestingInstances(test_inst)
        elif args.platform == "ipython":
            direct = True
            if args.direct == "false":
                direct = False
            if args.feature_algorithm == "color":
                bits = feature_converter.bits
                train_inst = ColorHistogram.ipython_par_color(training_files, bits, direct)
                test_inst = ColorHistogram.ipython_par_color(testing_files, bits, direct)
                feature_converter.setTrainingInstances(train_inst)
                feature_converter.setTestingInstances(test_inst)
            elif args.feature_algorithm == "bow":
                # TODO
                BagOfWords.BagOfWords(args.clusters)
            elif args.feature_algorithm == "img":
                feature_converter.createTrainingInstances(training_files)
                feature_converter.createTestingInstances(testing_files)
            elif args.feature_algorithm == "hog":
                train_inst = HOG.ipython_par_hog(training_files, direct)
                test_inst = HOG.ipython_par_hog(testing_files, direct)
                feature_converter.setTrainingInstances(train_inst)
                feature_converter.setTestingInstances(test_inst)
        else:
            raise Exception("Exception while attempting to run on platform")

        try:
            with open(args.feature_file, 'wb') as writer:
                pickle.dump(feature_converter, writer)
        except IOError:
            raise Exception("Exception while writing to the model file.")        
        except pickle.PickleError:
            raise Exception("Exception while dumping pickle.")

    elif mode == "train":
        # Load the training data.
        if args.feature_file is not None:
            feature_converter = load_data(args.feature_file)
        else:
            feature_converter = get_instance_converter(args.feature_algorithm, args)
            training_files = get_files(args.folder + '/train/', args.feature_algorithm, args)

            #parallize
            feature_converter.createTrainingInstances(training_files)

        # train some model
        # get model
        predictor = train(args.training_algorithm, args)
        #send to right methods for parallel
        predictor.train(feature_converter)
        # write_predictions(predictor, feature_converter, args.model_file)
        # save the model
        try:
            with open(args.model_file, 'wb') as writer:
                pickle.dump(predictor, writer)
        except IOError:
            raise Exception("Exception while writing to the model file.")        
        except pickle.PickleError as err:
            print err
            raise Exception("Exception while dumping pickle.")
            
    elif mode == "test":
        # Load the test data.
        if args.feature_file is not None:
            feature_converter = load_data(args.feature_file)
        else:
            feature_converter = get_instance_converter(args.feature_algorithm, args)
            testing_files = get_files(args.folder + '/test/', args.feature_algorithm, args)
            feature_converter.createTestingInstances(testing_files)

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
        write_predictions(predictor, feature_converter, args.predictions_file)
    else:
        raise Exception("Unrecognized mode.")

if __name__ == "__main__":
    main()
