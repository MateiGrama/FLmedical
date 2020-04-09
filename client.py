import torch
import torch.nn as nn
import torch.optim as optim


class Client:
    ''' An internal representation of a client '''

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
