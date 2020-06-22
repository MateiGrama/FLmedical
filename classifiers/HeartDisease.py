import torch.nn.functional as F
from torch import nn
from math import log, ceil


class Classifier(nn.Module):
    defaultInputSize = 14
    inputSize = None

    def __init__(self):
        super(Classifier, self).__init__()
        if not Classifier.inputSize:
            Classifier.inputSize = Classifier.defaultInputSize

        hiddenSize1 = pow(2, ceil(log(self.inputSize * 2) / log(2)))
        hiddenSize2 = pow(2, ceil(log(self.inputSize) / log(2)))

        self.fc1 = nn.Linear(self.inputSize, hiddenSize1)
        self.fc2 = nn.Linear(hiddenSize1, hiddenSize2)
        self.fc3 = nn.Linear(hiddenSize2, 2)

        self.drop_layer = nn.Dropout(p=0.5)

        Classifier.inputSize = None

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = self.drop_layer(x)
        x = F.relu(self.fc2(x))
        # x = self.drop_layer(x)
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=1)
        return x
