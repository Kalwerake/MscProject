import torch
import numpy as np


def train_one_epoch_binary(model, loss_fn, optimiser, train_loader, device):
    running_loss = 0
    epoch_accuracy = []
    for j, (x_train, y) in enumerate(train_loader):
        optimiser.zero_grad()  # zero the gradient at each epoch start
        y = y.to(device)  # send y to cuda
        x_train = x_train.to(device)
        prediction = model.forward(x_train)
        loss = loss_fn(prediction, y)  # loss

        accuracy = (torch.round(
            prediction) == y).float().mean()  # calculate accuracy for each mini-batch  take prediction tensor, reshape to 1d detach from computational graph turn to numpy array, round and see if rounded number is equal to label, find mean of this boolean array, this is the accuracy

        running_loss += loss.item()  # get epoch loss
        epoch_accuracy.append(accuracy.item())

        loss.backward()  # backward propgation
        optimiser.step()

        running_loss += loss.item()  # get epoch loss
        epoch_accuracy.append(accuracy.item())

    return running_loss, np.mean(epoch_accuracy)


def validate_one_epoch_binary(model, loss_fn, test_loader, device):
    test_loss_run = 0
    test_acc_epoch = []
    for j, (x_test, y_test) in enumerate(test_loader):
        y_test = y_test.to(device)
        x_test = x_test.to(device)
        test_pred = model.forward(x_test)
        test_loss = loss_fn(test_pred, y_test)  # loss

        test_acc = (torch.round(
            test_pred) == y_test).float().mean()  # calculate accuracy for each mini-batch  take prediction tensor, reshape to 1d detach from computational graph turn to numpy array, round and see if rounded number is equal to label, find mean of this boolean array, this is the accuracy

        test_loss_run += test_loss.item()
        test_acc_epoch.append(test_acc.item())

    return test_loss_run, np.mean(test_acc_epoch)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
