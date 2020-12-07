import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def iToSeqLength(i):
    if i <= 3:
        return 4
    elif i <= 6:
        return 5
    elif i <= 9:
        return 6
    else:
        return None

def moving_average(a, n=3):
    # Taken from https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_LSTM(type, basepath, title, output=None):

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Training iteration')
    ax1.set_ylabel('Accuracy')

    all_accuracies = None
    all_losses = None
    plots = []
    seq_lengths = [4*l for l in range(4, 6+1)]
    seeds = range(0, 2+1)
    for seq_len in seq_lengths:

        all_accuracies = None
        max_length = 0
        steps = None
        for seed in seeds:
            file = f"results_{type}_{seq_len}_{seed}.csv"
            df = pd.read_csv(os.path.join(basepath, file), index_col=0)
            if len(df) > max_length:
                max_length = len(df)
                steps = df.step.values
        print(steps)
        for seed in seeds:
            file = f"results_{type}_{seq_len}_{seed}.csv"
            df = pd.read_csv(os.path.join(basepath, file), index_col=0)

            val = df.accuracy.values
            padding_length = max_length - len(df)
            accuracy_pad = np.pad(val, (0, padding_length), 'constant', constant_values=val[-1])
            # print(pad)
            print(len(accuracy_pad))

            if all_accuracies is None:
                all_accuracies = accuracy_pad
            else:
                print(all_accuracies.shape)
                print(accuracy_pad.shape)
                all_accuracies = np.vstack((all_accuracies, accuracy_pad))
        
        mean = np.mean(all_accuracies, axis=0)
        std = np.std(all_accuracies, axis=0)
        l2 = ax1.plot(steps, mean, label=f"T={seq_len//4} accuracy")
        plots += l2
        ax1.fill_between(steps, mean-std, mean+std, alpha=0.5)
        all_accuracies = None
    plt.legend()
    plt.title(title)
    if output is not None:
        plt.savefig(os.path.join("images", output))
    plt.show()


def plot_LSTM(base_path, title, output=None):

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Training iteration')
    ax1.set_ylabel('Loss')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy')

    plots = []
    for i in range(2, 10, 3):
        path = f"{base_path}_{i}.csv"
        df = pd.read_csv(path, index_col=0)
        accuracy = moving_average(df.accuracy.values, n=10)
        loss = moving_average(df.loss.values, n=10)
        
       
        l2 = ax2.plot(range(len(accuracy)), accuracy, label=f"T={iToSeqLength(i)} accuracy")
        plots += l2
        l2 = ax1.plot(range(len(loss)), loss, label=f"T={iToSeqLength(i)} loss", linestyle="dotted")
        plots += l2

    labels = [plot.get_label() for plot in plots]
    plt.ylim(0.5)
    ax2.legend(plots, labels)
    plt.title(title)
    if output is not None:
        plt.savefig(os.path.join("images", output))
    plt.show()


# base_path = "../data/Q1.3_6940534"
base_path = "../data/Q1.4_6940487"
title = "Influence of the seqence length T on GRU performance"
output = "1.4-GRU3.png"
plot_LSTM(base_path, title, output)