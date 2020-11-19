"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

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
    
    class_preds = torch.argmax(predictions, dim=1)
    correct_preds = (class_preds == targets).sum()
    accuracy = correct_preds.item() / predictions.shape[0]
    
    return accuracy

def eval(model, dataset, batch_size, device):
    model.eval()

    total_images = 0
    total_accuracy = 0
    num_batches = 0
    while total_images < dataset.num_examples:
        images, labels = dataset.next_batch(batch_size)

        # Convert to torch Tensor
        images = torch.from_numpy(images).to(device)
        labels = torch.from_numpy(labels).to(device)

        preds = model.forward(images)
        batch_accuracy = accuracy(preds, labels)
        
        total_accuracy += batch_accuracy
        total_images += batch_size
        num_batches += 1
    
    return total_accuracy/num_batches

def train():
    """
    Performs training and evaluation of ConvNet model.
  
    TODO:
    Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
    """
    
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    ########################
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    print("Using device", device)

    def set_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available(): # GPU operation have separate seed
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    set_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cifar10 = cifar10_utils.get_cifar10(data_dir=FLAGS.data_dir, one_hot=False, validation_size=0)
    train = cifar10["train"]
    valid = cifar10["validation"]
    test = cifar10["test"]

    losses = []
    test_accuracies = []
    train_accuracies = []

    model = ConvNet(3, 10).to(device)
    loss_module = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
    
    step = 0 
    while step < FLAGS.max_steps:
        if step % FLAGS.eval_freq == 0: # Evaluate the model on the test dataset
            test_accuracy = eval(model, test, 100, device)
            test_accuracies.append(test_accuracy)
            print(f"STEP {step} - {test_accuracy}")
        
        model.train()
        images, labels = train.next_batch(FLAGS.batch_size)

        # Convert to torch Tensor
        images = torch.from_numpy(images).to(device)
        labels = torch.from_numpy(labels).to(device)

        preds = model(images)
        train_accuracies.append(accuracy(preds, labels))

        loss = loss_module(preds, labels)
        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
    
    test_accuracy = eval(model, test, 100, device)
    test_accuracies.append(test_accuracy)
    print(f"STEP {step} - {test_accuracy}")
    
    def moving_average(a, n=3):
        # Taken from https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    
    train_accuracies = moving_average(train_accuracies, n=100)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Training iteration')
    ax1.set_ylabel('Loss')
    l1 = ax1.plot(range(len(losses)), losses, label="training loss", color="b", alpha=0.5, linewidth=1)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy')
    l2 = ax2.plot(np.linspace(0, len(losses), len(test_accuracies)), test_accuracies, label="test accuracy", color="r")
    l3 = ax2.plot(np.linspace(0, len(losses), len(train_accuracies)), train_accuracies, label="train accuracy", color="b")

    plots = l1+l2+l3
    labels = [plot.get_label() for plot in plots]
    ax2.legend(plots, labels)

    plt.savefig(os.path.join("images", "cnn_loss_accuracy.png"))
    plt.show()


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
