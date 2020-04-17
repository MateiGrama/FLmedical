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
        self.score = alpha / betagit
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
    def __privacyPreserve(self, eps1=100, eps3=100, clipValue=0.1, releaseProportion=0.1,
                          needClip=False, needNormalization=False):
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
        noisyThreshold = 0  # laplace.rvs(scale=(s / e2)) + tau

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
                queryNoise = 0  # laplace.rvs(scale=(2 * shareParams * s / e1))
                if needClip:
                    noisyQuery = abs(clip(paramChange, -gamma, gamma)) + queryNoise
                else:
                    noisyQuery = abs(paramChange) + queryNoise

                if noisyQuery >= noisyThreshold:
                    answerNoise = 0  # laplace.rvs(scale=(shareParams * s / e3))
                    if needClip:
                        noisyChange = clip(paramChange + answerNoise, -gamma, gamma)
                    else:
                        noisyChange = paramChange + answerNoise

                    denormalisedChange = noisyChange * norm
                    modelNoisyParam = denormalisedChange + untrainedParams[paramName].data[index]
                    releasedParams[paramName][index] = modelNoisyParam
                    releaseParamsCount += 1

                    if releaseParamsCount < 5:
                        logPrint("Broadcast: {}\t"
                                 "Trained: {}\t"
                                 "Released: {}\t"
                                 "QueryNoise: {}\t"
                                 "AnswerNoise: {}\t"
                                 "".format(untrainedParams[paramName].data[index],
                                           params[paramName].data[index],
                                           releasedParams[paramName].data[index],
                                           round(queryNoise, 5),
                                           round(answerNoise, 5)))
                        sys.stdout.flush()

        # Update client model
        for paramName, param in params.items():
            param.data.copy_(releasedParams[paramName].data)

        logPrint("Privacy preserving for client{} in done.".format(self.id))

    # Procedure for implementing differential privacy
    def __optimPrivacyPreserve(self, eps1=100, eps3=100, clipValue=0.1, releaseProportion=0.1,
                               needClip=False, needNormalization=False):
        logPrint("Privacy preserving for client{} in process..".format(self.id))

        gamma = clipValue  # gradient clipping value
        s = 2 * gamma  # sensitivity
        Q = releaseProportion  # proportion to release

        params = dict(self.model.named_parameters())
        untrainedParams = dict(self.untrainedModel.named_parameters())

        # The gradients of the model parameters
        paramArr = array(nn.utils.parameters_to_vector(self.model.parameters()).detach())
        untrainedParamArr = array(nn.utils.parameters_to_vector(self.untrainedModel.parameters()).detach())
        paramChanges = paramArr - untrainedParamArr

        paramNo = len(paramArr)
        shareParamsNo = int(Q * paramNo)

        # Normalising
        if needNormalization:
            paramChanges /= self.n * self.epochs

        # List of (parameterName, parameterIndex=Tuple) for vectorized parameter operations
        paramIndex = []
        for paramName in params.keys():
            indexRanges = list(map(range, params[paramName].size()))
            possibleIndexes = product(*indexRanges)
            paramIndex += [(paramName, index) for index in possibleIndexes]
        paramIndex = array(paramIndex)

        # Privacy budgets for
        e1 = eps1  # gradient query
        e3 = eps3  # answer
        e2 = e1 * ((2 * shareParamsNo * s) ** (2 / 3))  # threshold

        tau = percentile(abs(paramChanges), Q * 100)
        noisyThreshold = laplace.rvs(scale=(s / e2)) + tau

        logPrint("NoisyThreshold: {}\t"
                 "e1: {}\t"
                 "e2: {}\t"
                 "e3: {}\t"
                 "shareParams: {}\t"
                 "".format(round(noisyThreshold, 3),
                           round(e1, 3),
                           round(e2, 3),
                           round(e3, 3),
                           round(shareParamsNo, 3),
                           ))

        r = torch.randperm(paramNo)
        paramChanges = paramChanges[r]
        paramIndex = paramIndex[r]

        queryNoise = [laplace.rvs(scale=(2 * shareParamsNo * s / e1)) for _ in range(paramNo)]
        # queryNoise = [0 for _ in range(paramNo)]  # )

        if needClip:
            noisyQuery = abs(clip(paramChanges, -gamma, gamma)) + queryNoise
        else:
            noisyQuery = abs(paramChanges) + queryNoise

        releaseIndex = noisyQuery >= noisyThreshold
        releaseChanges = paramChanges[releaseIndex][:shareParamsNo]
        releaseIndex = paramIndex[releaseIndex][:shareParamsNo]

        answerNoise = [laplace.rvs(scale=(shareParamsNo * s / e3)) for _ in range(shareParamsNo)]
        # answerNoise = [0 for _ in range(shareParamsNo)]  # )
        if needClip:
            noisyChanges = clip(releaseChanges + answerNoise, -gamma, gamma)
        else:
            noisyChanges = releaseChanges + answerNoise

        # Demoralising the noise
        if needNormalization:
            noisyChanges *= self.n * self.epochs

        name, index = releaseIndex[0]
        logPrint("Broadcast: {}\t"
                 "Trained: {}\t"
                 "Released: {}\t"
                 "ReleasedChange: {}\t"
                 "".format(untrainedParams[name].data[index],
                           params[name].data[index],
                           untrainedParams[name].data[index] + releaseChanges[0],
                           releaseChanges[0]))

        # Update client model
        for paramName, param in params.items():
            param.data.copy_(untrainedParams[paramName].data)

        i = 0
        for name, index in releaseIndex:
            params[name].data[index] += releaseChanges[i]
            i += 1

# TODO: (1) explore random seed examples; (2) make index map and concatenate all names?
#       (3) try out different configurations params configs; plot them;
#       (4)


# In the future:
# different number of epochs for different clients
