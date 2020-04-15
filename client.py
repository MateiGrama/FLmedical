import sys
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim

from numpy import clip, percentile
from scipy.stats import laplace

from logger import logPrint


class Client:
    """ An internal representation of a client """

    def __init__(self, model, epochs, batchSize, x, y, p, idx, byzantine, flip, device,
                 alpha=3.0, beta=3.0):
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

        # Used for computing dW, i.e. the change in model before
        # and after client local training, when DP is used
        self.untrainedModel = copy.deepcopy(model).to(self.device) if model else False

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
        self.untrainedModel = copy.deepcopy(model).to(self.device)

    # Function to train the model for a specific user
    def trainModel(self):
        for i in range(self.epochs):
            # logPrint("Epoch user: ",i)
            # Shuffle training data
            r = torch.randperm(self.xTrain.size()[0])
            self.xTrain = self.xTrain[r]
            self.yTrain = self.yTrain[r]
            for iBatch in range(0, self.xTrain.size(0), self.batchSize):
                x = self.xTrain[iBatch:iBatch + self.batchSize, :]
                y = self.yTrain[iBatch:iBatch + self.batchSize]
                err, pred = self._trainClassifier(x, y)
        return err, pred

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

    # Function used by aggregators to retrieve the model from the client
    def retrieveModel(self, differentialPrivacy=False):
        if self.byz:
            # Malicious model update
            # logPrint("Malicous update for user ",u.id)
            self.__manipulateModel()

        if differentialPrivacy:
            self.__privacyPreserve()

        return self.model

    # Function to manipulate the model for byzantine adversaries
    def __manipulateModel(self, alpha=20):
        params = self.model.named_parameters()
        for name, param in params:
            noise = alpha * torch.randn(param.data.size()).to(self.device)
            param.data.copy_(param.data + noise)

    # Procedure for implementing differential privacy
    def __privacyPreserve(self, eps1=1, eps3=1, clipValue=0.0001, releaseProportion=0.1,
                          needClip=True, needNormalization=False):
        logPrint("Privacy preserving for client{} in process..".format(self.id))

        gamma = clipValue  # gradient clipping value
        s = 2 * gamma  # sensitivity
        Q = releaseProportion  # proportion to release

        norm = self.n * self.epochs if needNormalization else 1
        paramNo = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        shareParams = Q * paramNo

        # Privacy budgets for
        e1 = eps1  # gradient query
        e3 = eps3  # answer
        e2 = e1 * ((2 * shareParams * s) ** (2 / 3))  # threshold

        paramTuples = zip(nn.utils.parameters_to_vector(self.model.parameters()),
                          nn.utils.parameters_to_vector(self.untrainedModel.parameters()))
        paramChangesArr = [abs(param.data - untrainedParam.data) for param, untrainedParam in paramTuples]

        tau = percentile(paramChangesArr, Q * 100)
        noisyThreshold = laplace.rvs(scale=(s / e2)) + tau

        logPrint("NoisyThreshold: {}\t"
                 "e1: {}\t"
                 "e2: {}\t"
                 "e3: {}\t"
                 "shareParams: {}\t"
                 "paramNo: {}\t"
                 "".format(round(noisyThreshold, 3),
                           round(e1, 3),
                           round(e2, 3),
                           round(e3, 3),
                           round(shareParams, 3),
                           round(paramNo, 3),
                           ))

        params = dict(self.model.named_parameters())
        untrainedParams = dict(self.untrainedModel.named_parameters())
        changeOfParams = dict()
        releasedParams = dict()

        for paramName, param in params.items():
            # Compute params change normalised by iterations: dW = dW/N_local
            paramChange = param.data - untrainedParams[paramName].data
            normalised = paramChange / norm
            changeOfParams[paramName] = normalised

            # Initialize release parameters accumulator
            releasedParams[paramName] = torch.clone(untrainedParams[paramName].data)

        releaseParamsCount = 0
        while releaseParamsCount < shareParams:
            #  Randomly draw a gradient component by selecting
            #  a random param and an random index of the param
            paramName = random.choice(list(params.keys()))
            paramSize = torch.tensor(params[paramName].size())
            index = tuple((torch.rand(paramSize.size()) * (paramSize - 1)).to(torch.long))

            # If not already selected for realise (i.e. value of accumulator param still equal
            # to the aggregator broadcast model) proceed to check if above noisy threshold
            paramNotReleasedYet = (releasedParams[paramName][index] == untrainedParams[paramName].data[index])
            if paramNotReleasedYet:
                paramChange = changeOfParams[paramName][index]
                queryNoise = laplace.rvs(scale=(2 * shareParams * s / e1))
                if needClip:
                    noisyQuery = abs(clip(paramChange, -gamma, gamma)) + queryNoise
                else:
                    noisyQuery = abs(paramChange) + queryNoise

                if noisyQuery >= noisyThreshold:
                    answerNoise = laplace.rvs(scale=(shareParams * s / e3))
                    if needClip:
                        noisyChange = clip(paramChange + answerNoise, -gamma, gamma)
                    else:
                        noisyChange = paramChange + answerNoise

                    denormalisedChange = noisyChange * norm
                    modelNoisyParam = denormalisedChange + untrainedParams[paramName].data[index]
                    releasedParams[paramName][index] = modelNoisyParam
                    releaseParamsCount += 1

                    if releaseParamsCount < 5:
                        logPrint("Agg val: {}\t"
                                 "Client model: {}\t"
                                 "Release: {}\t"
                                 "Query Noise: {}\t"
                                 "Answer Noise: {}\t"
                                 "".format(untrainedParams[paramName].data[index],
                                           params[paramName].data[index],
                                           releasedParams[paramName].data[index],
                                           round(queryNoise, 3),
                                           round(answerNoise, 3)))
                        sys.stdout.flush()

        # Update client model
        for paramName, param in params.items():
            param.data.copy_(releasedParams[paramName].data)

        logPrint("Privacy preserving for client{} in done.".format(self.id))

    # In the future:
    # different number of epochs for different clients

# In the future:
# different number of epochs for different clients
