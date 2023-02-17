import torch
import pandas as pd
import pickle

from nn_functions import train_one_epoch_binary, validate_one_epoch_binary, EarlyStopper
from models import CRNN5
from custom_dataset import SurrDataset

from torch.utils.data import DataLoader
import torch.optim as optim

import torch.nn as nn
import argparse
import pathlib


def main(train_path, test_path, data_dir, batch_size, lr, epochs, workers, model_save,
         metric_path, early_stop=False, patience=0, min_delta=0):
    """

    :param train_path:
    :param test_path:
    :param data_dir:
    :param batch_size:
    :param lr:
    :param epochs:
    :param workers:
    :param model_save: save path for saving model
    :param metric_path: path for saving model_metric
    :param early_stop:
    :param min_delta:
    :param patience:
    :return:
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_data = SurrDataset(train_df, data_dir)
    train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=workers)

    test_data = SurrDataset(test_df, data_dir)
    test_dataloader = DataLoader(test_data, batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CRNN5()
    loss_fn = nn.BCELoss()
    optimiser = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    model.to(device)

    train_loss_history = []
    train_acc_history = []
    test_loss_history = []
    test_acc_history = []

    if early_stop:
        early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
    else:
        pass

    for i in range(epochs):

        model.train()
        train_loss, train_acc = train_one_epoch_binary(model, loss_fn, optimiser, train_dataloader, device)

        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)

        model.eval()
        test_loss, test_acc = validate_one_epoch_binary(model, loss_fn, test_dataloader, device)

        test_loss_history.append(test_loss)
        test_acc_history.append(test_acc)

        if early_stop and early_stopper.early_stop(test_loss):
            break

    torch.save(model.state_dict(), model_save)

    metrics = {'train_acc_history': train_acc_history, 'train_loss_history': train_loss_history,
               'test_acc': test_acc_history, 'test_loss': test_loss_history}  # make dictionary of metrics

    with open(metric_path, 'wb') as handle:
        pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Train',
        description='train, save and evaluate model')

    parser.add_argument('--train', help='path to train csv', type=pathlib.Path)
    parser.add_argument('--test', help='path to test csv', type=pathlib.Path)
    parser.add_argument('--data', help='path to data directory', type=pathlib.Path)
    parser.add_argument('--batch', help='batch size', type=int)
    parser.add_argument('--lr', help='learning rate', type=float)
    parser.add_argument('--epochs', help='epoch number', type=int)
    parser.add_argument('--workers', help='num workers for data loader', type=int)
    parser.add_argument('--model_save', help='path to save model parameters', type=pathlib.Path)
    parser.add_argument('--metric_save', help='path to save model metrics', type=pathlib.Path)

    parser.add_argument('--early_stop', help='enact early stopping?', action=argparse.BooleanOptionalAction)
    parser.add_argument('--patience', help='patience', type=int, required=False)
    parser.add_argument('--delta', help='minimum change in validation loss', type=float, required=False)

    args = parser.parse_args()

    main(train_path=args.train, test_path=args.test, data_dir=args.data, batch_size=args.batch, lr=args.lr,
         epochs=args.epochs, workers=args.workers, model_save=args.model_save, metric_path=args.metric_save,
         early_stop=args.early_stop, patience=args.patience, min_delta=args.delta)
