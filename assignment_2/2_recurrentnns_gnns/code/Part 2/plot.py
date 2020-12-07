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

def plot_LSTM(base_path, title, output=None):

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Training iteration')
    ax1.set_ylabel('Loss')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy')

    all_accuracies = None
    all_losses = None
    plots = []
    for i in range(1, 10):
        if i in [1, 4, 7]:
            print(i)
            min_length = 3000
            for j in range(3):
                path = f"{base_path}_{i+j}.csv"
                df = pd.read_csv(path, index_col=0)
                if len(df) < min_length: 
                    min_length = len(df)
            print(min_length)
        path = f"{base_path}_{i}.csv"
        df = pd.read_csv(path, index_col=0)
        accuracy = df.accuracy.values[:min_length]
        print(accuracy.shape)
        loss = df.loss.values[:min_length]
        
        if all_accuracies is None:
            all_accuracies = accuracy
        else:
            print(all_accuracies.shape)
            print(accuracy.shape)
            all_accuracies = np.vstack((all_accuracies, accuracy))
        
        if all_losses is None:
            all_losses = loss
        else:
            all_losses = np.vstack((all_losses, loss))
        
        if i % 3 == 0:
            # Accuracy
            mean = moving_average(np.mean(all_accuracies, axis=0), n=20)[:1500]
            std = moving_average(np.std(all_accuracies, axis=0), n=20)[:1500]
            l2 = ax2.plot(range(len(mean)), mean, label=f"T={iToSeqLength(i)} accuracy")
            plots += l2
            ax2.fill_between(range(len(mean)), mean-std, mean+std, alpha=0.5)
            all_accuracies = None

            # Loss
            mean = moving_average(np.mean(all_losses, axis=0), n=20)[:1500]
            std = moving_average(np.std(all_losses, axis=0), n=20)[:1500]
            l2 = ax1.plot(range(len(mean)), mean, label=f"T={iToSeqLength(i)} loss", linestyle="dashed")
            plots += l2
            all_accuracies = None

    labels = [plot.get_label() for plot in plots]
    ax2.legend(plots, labels)
    plt.title(title)
    if output is not None:
        plt.savefig(os.path.join("images", output))
    plt.show()

# base_path = "../data/Q1.3_6857498"
# title = "Influence of the seqence length T on an LSTM performance"
# output = "1.3-LSTM.png"
# plot_LSTM(base_path, title, None)

# base_path = "../data/Q1.4_6856484"
# title = "Influence of the seqence length T on a GRU performance"
# output = "1.4-GRU.png"
# plot_LSTM(base_path, title, output)

def plot_LSTM(book, basepath, title, output=None):

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Training iteration')
    ax1.set_ylabel('Loss')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy')

    all_accuracies = None
    all_losses = None
    plots = []
    seeds = range(0, 2+1)

    all_accuracies = None
    max_length = 0

    for seed in seeds:
        file = f"results_{book}_{seed}.csv"
        df = pd.read_csv(os.path.join(basepath, file), index_col=0)

        accuracy = moving_average(df.accuracy.values, n=100)
        loss = moving_average(df.loss.values, n=100)
        
        l2 = ax2.plot(range(len(accuracy)), accuracy, label="Accuracy", linewidth=2)
        plots += l2
        l1 = ax1.plot(range(len(loss)), loss, label="Loss", linestyle="dotted", linewidth=2)
        plots += l1
        all_accuracies = None
        break
    
    labs = [l.get_label() for l in plots]
    plt.legend(plots, labs)
    # plt.xlim(0, 10000)
    # plt.legend()
    plt.title(title)
    if output is not None:
        plt.savefig(os.path.join("images", output))
    plt.show()


base_path = "../../data"
title = "Grimms"
output = "2.1a-grimms.png"
# name = "US_6879910"
name = "grimms_6879913"
plot_LSTM(name, base_path, title, output)