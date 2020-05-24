from torch import nn


class Classifier(nn.Module):
    inputSize = 8

    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(self.inputSize, 200)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(200, 200)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(200, 2)
        self.out_act = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.out(x)
        x = self.out_act(x)
        return x
