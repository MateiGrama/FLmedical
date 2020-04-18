import sys
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product
from numpy import clip, percentile, array, concatenate, empty

from scipy.stats import laplace

from logger import logPrint


class Client:
    """ An internal representation of a client """

    def __init__(self, epochs, batchSize, x, y, p, idx, device, useDifferentialPrivacy,
                 releaseProportion, epsilon1, epsilon3, needClip, clipValue, needNormalization,
                 byzantine=None, flipping=None, model=None, alpha=3.0, beta=3.0):

        self.name = "client" + str(idx)
        self.device = device

        self.model = model
        self.xTrain = x  # Training data
        self.yTrain = y  # Labels
        self.p = p  # Contribution to the overall model
        self.n = x.size()[0]  # Number of training points provided
        self.id = idx  # ID for the user
        self.byz = byzantine  # Boolean indicating whether the user is faulty or not
        self.flip = flipping  # Boolean indicating whether the user is malicious or not (label flipping attack)

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

        # AFA Client params
        self.alpha = alpha
        self.beta = beta
        self.score = alpha / beta
        self.blocked = False

        # DP parameters
        self.useDifferentialPrivacy = useDifferentialPrivacy
        self.epsilon1 = epsilon1
        self.epsilon3 = epsilon3
        self.needClip = needClip
        self.clipValue = clipValue
        self.needNormalization = needNormalization
        self.releaseProportion = releaseProportion

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
                x = self.xTrain[iBatch:iBatch + self.batchSize, :].to(self.device)
                y = self.yTrain[iBatch:iBatch + self.batchSize].to(self.device)
                err, pred = self._trainClassifier(x, y)
        return err, pred

    # Function to train the classifier
    def _trainClassifier(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        # Reset gradients
        self.opt.zero_grad()
        pred = self.model(x).to(self.device)
        err = self.loss(pred, y).to(self.device)
        err.backward()
        # Update optimizer
        self.opt.step()
        return err, pred

    # Function used by aggregators to retrieve the model from the client
    def retrieveModel(self):
        if self.byz:
            # Malicious model update
            # logPrint("Malicous update for user ",u.id)
            self.__manipulateModel()

        if self.useDifferentialPrivacy:
            # self.__privacyPreserve()
            self.__optimPrivacyPreserve()
        return self.model

    # Function to manipulate the model for byzantine adversaries
    def __manipulateModel(self, alpha=20):
        params = self.model.named_parameters()
        for name, param in params:
            noise = alpha * torch.randn(param.data.size()).to(self.device)
            param.data.copy_(param.data + noise)

    # Procedure for implementing differential privacy
    def __optimPrivacyPreserve(self, eps1=100, eps3=100, clipValue=0.1, releaseProportion=0.1,
                               needClip=False, needNormalization=False):
        # logPrint("Privacy preserving for client{} in process..".format(self.id))


        gamma = clipValue # gradient clipping value
        s = 2 * gamma  # sensitivity
        Q = releaseProportion  # proportion to release

        # The gradients of the model parameters
        paramArr = nn.utils.parameters_to_vector(self.model.parameters())
        untrainedParamArr = nn.utils.parameters_to_vector(self.untrainedModel.parameters())

        paramNo = len(paramArr)
        shareParamsNo = int(Q * paramNo)

        r = torch.randperm(paramNo).to(self.device)
        paramArr = paramArr[r].to(self.device)
        untrainedParamArr = untrainedParamArr[r].to(self.device)
        paramChanges = (paramArr - untrainedParamArr).detach().to(self.device)

        # Normalising
        if needNormalization:
            paramChanges /= self.n * self.epochs

        # Privacy budgets for
        e1 = eps1  # gradient query
        e3 = eps3  # answer
        e2 = e1 * ((2 * shareParamsNo * s) ** (2 / 3))  # threshold

        paramChanges = paramChanges.cpu()
        tau = percentile(abs(paramChanges), Q * 100)
        paramChanges = paramChanges.to(self.device)
        # tau = 0.0001
        noisyThreshold = laplace.rvs(scale=(s / e2)) + tau

        # logPrint("NoisyThreshold: {}\t"
        #          "e1: {}\t"
        #          "e2: {}\t"
        #          "e3: {}\t"
        #          "shareParams: {}\t"
        #          "".format(round(noisyThreshold, 3),
        #                    round(e1, 3),
        #                    round(e2, 3),
        #                    round(e3, 3),
        #                    round(shareParamsNo, 3),
        #                    ))

        queryNoise = laplace.rvs(scale=(2 * shareParamsNo * s / e1), size=paramNo)
        queryNoise = torch.tensor(queryNoise).to(self.device)
        # queryNoise = [0 for _ in range(paramNo)]  # )

        if needClip:
            noisyQuery = abs(clip(paramChanges, -gamma, gamma)) + queryNoise
        else:
            noisyQuery = abs(paramChanges) + queryNoise
        noisyQuery = noisyQuery.to(self.device)

        releaseIndex = (noisyQuery >= noisyThreshold).to(self.device)
        filteredChanges = paramChanges[releaseIndex]

        answerNoise = laplace.rvs(scale=(shareParamsNo * s / e3), size=torch.sum(releaseIndex).cpu())
        answerNoise = torch.tensor(answerNoise).to(self.device)
        if needClip:
            noisyFilteredChanges = clip(filteredChanges + answerNoise, -gamma, gamma)
        else:
            noisyFilteredChanges = filteredChanges + answerNoise
        noisyFilteredChanges = noisyFilteredChanges.to(self.device)

        # Demoralising the noise
        if needNormalization:
            noisyFilteredChanges *= self.n * self.epochs

        # logPrint("Broadcast: {}\t"
        #          "Trained: {}\t"
        #          "Released: {}\t"
        #          "answerNoise: {}\t"
        #          "ReleasedChange: {}\t"
        #          "".format(untrainedParamArr[releaseIndex][0],
        #                    paramArr[releaseIndex][0],
        #                    untrainedParamArr[releaseIndex][0] + noisyFilteredChanges[0],
        #                    answerNoise[0],
        #                    noisyFilteredChanges[0]))
        # sys.stdout.flush()

        paramArr = untrainedParamArr
        paramArr[releaseIndex][:shareParamsNo] += noisyFilteredChanges[:shareParamsNo]
        # logPrint("Privacy preserving for client{} done.".format(self.id))


# In the future:
# different number of epochs for different clients
