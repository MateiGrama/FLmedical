import torch
import torch.nn as nn
import torch.optim as optim

from numpy import clip, percentile
from scipy.stats import laplace


class Client:
    """ An internal representation of a client """

    def __init__(self, model, epochs, batchSize, x, y, p, idx, byzantine, flip, device,
                 alpha=3, beta=3):
        self.name = "client" + str(idx)
        self.device = device

        self.model = model
        self.xTrain = x  # Training data
        self.yTrain = y  # Labels
        self.p = p  # Contribution to the overall model
        self.n = x.size()[0]  # Number of training points provided
        self.id = idx  # ID for the user
        self.byz = byzantine  # Boolean indicating whether the user is faulty or not
        self.flip = flip  # Boolean indicating whether the user is malicious or not (label flipping attack)

        self.opt = None
        self.sim = None
        self.loss = None
        self.pEpoch = None
        self.badUpdate = False
        self.epochs = epochs
        self.batchSize = batchSize

        self.learningRate = 0.1
        self.momentum = 0.9

        # FA Client params
        self.alpha = alpha
        self.beta = beta
        self.score = alpha / beta
        self.blocked = False

    def updateModel(self, model):
        self.model = model
        self.opt = optim.SGD(self.model.parameters(), lr=self.learningRate, momentum=self.momentum)
        # u.opt = optim.Adam(u.model.parameters(), lr=0.001)
        self.loss = nn.CrossEntropyLoss()

    # Function to train the classifier
    def _trainClassifier(self, x, y):
        x.to(self.device)
        y.to(self.device)
        # Reset gradients
        self.opt.zero_grad()
        pred = self.model(x)
        pred.to(self.device)
        err = self.loss(pred, y).to(self.device)
        err.backward()
        # Update optimizer
        self.opt.step()
        return err, pred

    # Function to train the model for a specific user
    def trainModel(self):
        for i in range(self.epochs):
            # print("Epoch user: ",i)
            # Shuffle training data
            r = torch.randperm(self.xTrain.size()[0])
            self.xTrain = self.xTrain[r]
            self.yTrain = self.yTrain[r]
            for iBatch in range(0, self.xTrain.size(0), self.batchSize):
                x = self.xTrain[iBatch:iBatch + self.batchSize, :]
                y = self.yTrain[iBatch:iBatch + self.batchSize]
                err, pred = self._trainClassifier(x, y)
        return err, pred

    # Function to manipulate the model for byzantine adversaries
    def manipulateModel(self, alpha=20):
        params = self.model.named_parameters()
        for name, param in params:
            noise = alpha * torch.randn(param.data.size()).to(self.device)
            param.data.copy_(param.data + noise)

    # Procedure for implementing differential privacy
    def privacyPreserve(self, eps1=0.001, eps3=0.001, clipValue=0.001, releaseProportion=0.1):

        gamma = clipValue  # gradient clipping value
        s = 2 * gamma      # sensitivity
        Q = releaseProportion  # proportion to release

        paramArray = [abs(p.data) for p in nn.utils.parameters_to_vector(self.model.parameters())]
        tau = percentile(paramArray, Q * 100)

        paramNo = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        shareParams = Q * paramNo

        # Privacy budgets for
        e1 = eps1                                 # gradient query
        e2 = e1 * (2 * shareParams * s) ** (2/3)  # threshold
        e3 = eps3                                 # answer

        noisyThreshold = laplace(s / e2) + tau
        queryThreshold = laplace(2 * shareParams * s / e1)
        answerNoise    = laplace(shareParams * s / e3)

        # Normalise by iterations
        params = self.model.named_parameters()
        for _, param in params:
            normalised = param.data / (self.epochs * self.n)
            param.data.copy_(normalised)

        releaseParams = dict()
        while len(releaseParams) < shareParams:
            #  Randomly draw a gradient component
            unreleasedNames = set(params.keys()).difference(releaseParams.keys())
            paramName = unreleasedNames.pop()

            if abs(clip(params[paramName].data, -gamma, gamma)) + queryThreshold >= noisyThreshold:
                noisyAnswer = clip(params[paramName].data + answerNoise, -gamma, gamma)
                releaseParams[paramName] = noisyAnswer

        # Denormalise
        for _, param in params:
            normalised = param.data * (self.epochs * self.n)
            param.data.copy_(normalised)

# Issues:
#   while loop may loop forever
#   what is tau
#   0ing not released parameters
#   Nlocal of the aggregation algorithm: epochs or data


# In the future:
# different number of epochs for different clients

