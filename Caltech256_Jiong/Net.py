import torch.nn as nn
import torch.nn.functional as F


class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=1)
        self.fc1   = nn.Linear( 6144,4096)
        self.fc2   = nn.Linear( 4096,4096)
        self.fc3   = nn.Linear( 4096,10)
        self.dropout = nn.Dropout(0.25)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # forward
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), 6144 )
        nn.Dropout()
        x = F.relu(self.fc1(x))
        nn.Dropout()
        x = F.relu(self.fc2(x))
        nn.Dropout()
        x = F.relu(self.fc3(x))
        x = self.softmax(x)
        return x
