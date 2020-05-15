from torch import nn


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(8, 4)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(4, 3)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(3, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.out(x)
        x = self.out_act(x)
        return x
