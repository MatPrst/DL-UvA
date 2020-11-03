"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 1400 #1400
BATCH_SIZE_DEFAULT = 200 #200
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.02

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.
    
    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch
    
    TODO:
    Implement accuracy computation.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################
    
    return accuracy


def train():
    """
    Performs training and evaluation of MLP model.
  
    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """
    
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []
    
    # neg_slope = FLAGS.neg_slope
    
    ########################
    cifar10 = cifar10_utils.get_cifar10(data_dir=DATA_DIR_DEFAULT, one_hot=False, validation_size=0)
    train = cifar10["train"]
    valid = cifar10["validation"]
    test = cifar10["test"]

    model = MLP(3*32*32, dnn_hidden_units, 10)
    loss_module = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=FLAGS.learning_rate)
    
    model.train()
    step = 0 
    while step < FLAGS.max_steps:
        if step % FLAGS.eval_freq == 0:

            images, labels = test.next_batch(10000)

            # Reshape and convert to torch Tensor
            images = torch.from_numpy(images.reshape(images.shape[0], -1))
            labels = torch.from_numpy(labels)

            preds = model(images)

            class_preds = torch.argmax(preds, dim=1)
            correct_preds = (class_preds == labels).sum().numpy()
            accuracy_batch = correct_preds / labels.shape[0]
            print(f"STEP {step} - {accuracy_batch}")
        
        images, labels = train.next_batch(FLAGS.batch_size)

        # Reshape and convert to torch Tensor
        images = torch.from_numpy(images.reshape(images.shape[0], -1))
        labels = torch.from_numpy(labels)

        preds = model(images)

        loss = loss_module(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()
    
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)
    
    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    
    main()
