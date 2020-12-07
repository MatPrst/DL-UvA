from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel

def get_model_file(epoch, config):
    name = f"model_{epoch}_{config.seed}.tar"
    return name

def sample(model, dataset, init_seq, init_hidden, seq_length, device, temp=None):
    model.eval()
    batch_preds, hidden = model(init_seq, init_hidden)
    
    s = ""
    for _ in range(seq_length):
        if temp is None:
            pred = torch.argmax(batch_preds, dim=2)
        else:
            pred = torch.multinomial(torch.softmax(temp * batch_preds, dim=2)[0], 1)

        pred_token = dataset.convert_to_string(pred.data.cpu().numpy().flatten())
        
        s += pred_token
        batch_preds, hidden = model(pred, hidden)
    return s

def generate(config):
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)  # fixme
    data_loader = DataLoader(dataset, config.batch_size, drop_last=True)

    # Initialize the model that we are going to use
    model = TextGenerationModel(
        batch_size=config.batch_size,
        seq_length=config.seq_length,
        vocabulary_size=86,
        lstm_num_hidden=config.lstm_num_hidden,
        lstm_num_layers=config.lstm_num_layers,
        device=config.device
        ).to(device)
    model.load_state_dict(torch.load(config.model))


    for l in ["In 1776 ", "Liberty is ", "Democracy is "]:
        char_id = torch.tensor([dataset._char_to_ix[ch] for ch in l]).reshape(-1, 1).to(device)
        hidden = (
            torch.zeros((config.lstm_num_layers, 1, config.lstm_num_hidden)).to(device),
            torch.zeros((config.lstm_num_layers, 1, config.lstm_num_hidden)).to(device)
        )
        sequence = sample(
            model=model, 
            dataset=dataset, 
            init_seq=char_id, 
            init_hidden=hidden, 
            seq_length=200, 
            device=device,
            temp=config.temp)
        print(dataset.convert_to_string(char_id.cpu().numpy().reshape(-1)) + sequence)


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True,
                        help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30,
                        help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128,
                        help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2,
                        help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3,
                        help='Learning rate')

    # It is not necessary to implement the following three params,
    # but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96,
                        help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000,
                        help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0,
                        help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=int(1e6),
                        help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/",
                        help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5,
                        help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100,
                        help='How often to sample from the model')
    parser.add_argument('--device', type=str, default=("cpu" if not torch.cuda.is_available() else "cuda"),
                        help="Device to run the model on.")

    # If needed/wanted, feel free to add more arguments
    parser.add_argument('--output', type=str, default="./results.csv",
                        help='Path to the csv output file containing the results')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs')
    parser.add_argument('--temp', type=float, default=1.0,
                        help='Temperature of the sampling process')
    parser.add_argument('--model', type=str, required=True, 
                        help='Path to the trained model')

    config = parser.parse_args()

    generate(config)