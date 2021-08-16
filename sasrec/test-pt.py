import torch
import time

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

if __name__ == '__main__':
    device = 'cuda'
    x = torch.randn((128, 200, 50)).to(device)
    layer = PointWiseFeedForward(50, 0.5)
    layer = layer.to(device)
    for i in range(100):
        t0 = time.time()
        for j in range(200):
            # z = x.transpose(0, 1)
            # y = layer(z, z, z)
            y = layer(x)
        print(time.time() - t0)

