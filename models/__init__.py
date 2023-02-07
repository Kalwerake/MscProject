import torch.nn as nn
import torch


class CRNN5(nn.Module):
    def __init__(self):
        super(CRNN5, self).__init__()

        # Convolutional layers
        self.seq_cnn = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(2, 200, 1), stride=(1, 1, 1), padding_mode='reflect'),
            nn.GELU(),
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(2, 1, 200), stride=(1, 1, 1),
                      padding_mode='reflect'),
            nn.GELU(),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(8, 1, 1), stride=(2, 1, 1), padding_mode='reflect'),
            nn.GELU(),
            nn.BatchNorm3d(32)
        )

        # lstm layer
        self.lstm1 = nn.GRU(input_size=32, hidden_size=64, num_layers=1, batch_first=True)

        # Fully connected layer
        self.seq_dense = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.seq_cnn(x)

        x = torch.flatten(x, start_dim=2, end_dim=4)  # reduce dimensionality for LSTM layer, to 3D tensor
        x = x.permute(0, 2, 1)  # transpose to make tensor of size [batch_size, sequence length, feature number]

        x, _ = self.lstm1(x)

        x = x[:, -1, :]  # take output of last LSTM cell

        x = self.seq_dense(x)

        return x
