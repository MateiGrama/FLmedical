import random

import numpy as np
import torch
import matplotlib.pyplot as plt

from aggregator import aggregators
from dataUtils import loadMNISTdata
from client import Client
import classifierMNIST
from logger import logPrint

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classifier = classifierMNIST.Classifier

# TRAINING PARAMETERS
rounds = 10  # TOTAL NUMBER OF TRAINING ROUNDS
epochs = 10  # NUMBER OF EPOCHS RUN IN EACH CLIENT BEFORE SENDING BACK THE MODEL UPDATE
batch_size = 200  # BATCH SIZE


def trainOnMNIST(aggregator, perc_users, labels, faulty, flipping, privacyPreserving=False):
    logPrint("Loading MNIST...")
    training_data, training_labels, xTest, yTest = loadMNISTdata(perc_users, labels)

    clients = initClients(perc_users, training_data, training_labels, faulty, flipping)

    # CREATE MODEL
    model = classifier().to(device)
    aggregator = aggregator(clients, model, rounds, device,
                            useDifferentialPrivacy=privacyPreserving)
    return aggregator.trainAndTest(xTest, yTest)


def initClients(perc_users, training_data, training_labels, faulty, flipping):
    usersNo = perc_users.size(0)
    p0 = 1 / usersNo
    # Seed
    __setRandomSeeds(2)
    logPrint("Creating clients...")
    clients = []
    for i in range(usersNo):
        clients.append(Client(model=False,
                              epochs=epochs,
                              batchSize=batch_size,
                              x=training_data[i],
                              y=training_labels[i],
                              p=p0,
                              idx=i + 1,
                              byzantine=False,
                              flip=False,
                              device=device))

    ntr = 0
    for client in clients:
        ntr += client.xTrain.size(0)

    # Weight the value of the update of each user according to the number of training data points
    for client in clients:
        client.p = client.xTrain.size(0) / ntr
        # logPrint("Weight for user ", u.id, ": ", round(u.p,3))

    # Create malicious (byzantine) users
    for client in clients:
        if client.id in faulty:
            client.byz = True
            logPrint("User ", client.id, " is faulty.")
        if client.id in flipping:
            client.flip = True
            logPrint("User ", client.id, " is malicious.")
            # Flip labels
            # r = torch.randperm(u.ytr.size(0))
            # u.yTrain = u.yTrain[r]
            client.yTrain = torch.zeros(client.yTrain.size(), dtype=torch.int64)
    return clients


def __setRandomSeeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # cudnn.deterministic = True
    # warnings.warn('You have chosen to seed training. '
    #               'This will turn on the CUDNN deterministic setting, '
    #               'which can slow down your training considerably! '
    #               'You may see unexpected behavior when restarting '
    #               'from checkpoints.')


# EXPERIMENTS #

def noByzClientMNISTExperiment():
    perc_users = torch.tensor([0.2, 0.10, 0.15, 0.15, 0.15, 0.15, 0.1])
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    faulty = []
    malicious = []
    _testAggregators(perc_users, labels, faulty, malicious)


def byzClientMNISTExperiment():
    perc_users = torch.tensor([0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1])
    labels = torch.tensor([0, 2, 5, 8])
    faulty = [2, 6]
    malicious = [1]
    _testAggregators(perc_users, labels, faulty, malicious)


def privacyPreservingNoByzClientMNISTExperiment():
    perc_users = torch.tensor([0.2, 0.10, 0.15, 0.15, 0.15, 0.15, 0.1])
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    faulty = []
    malicious = []
    _testPrivacyPreservingAggregators(perc_users, labels, faulty, malicious)


def privacyPreservingAndVanillaNoByzClientMNIST():
    perc_users = torch.tensor([0.2, 0.10, 0.15, 0.15, 0.15, 0.15, 0.1])
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    faulty = []
    malicious = []
    _testBothAggregators(perc_users, labels, faulty, malicious)


def privacyPreservingByzClientMNISTExperiment():
    perc_users = torch.tensor([0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1])
    labels = torch.tensor([0, 2, 5, 8])
    faulty = [2, 6]
    malicious = [1]
    _testPrivacyPreservingAggregators(perc_users, labels, faulty, malicious)


def privacyPreservingFewClientsMNISTExperiment():
    perc_users = torch.tensor([0.3, 0.25, 0.45])
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    faulty = []
    malicious = []
    _testPrivacyPreservingAggregators(perc_users, labels, faulty, malicious)


def noDP30ClientsMNISTExperiment():
    perc_users = torch.tensor([0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2,
                               0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2,
                               0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2,
                               ])
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    faulty = []
    malicious = []
    _testAggregators(perc_users, labels, faulty, malicious)


def privacyPreserving30ClientsMNISTExperiment():
    perc_users = torch.tensor([0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2,
                               0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2,
                               0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2,
                               ])
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    faulty = []
    malicious = []
    _testPrivacyPreservingAggregators(perc_users, labels, faulty, malicious)


def testBothAgg30ClientsMNISTExperiment():
    perc_users = torch.tensor([0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2,
                               0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2,
                               0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2,
                               ])
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    faulty = []
    malicious = []
    _testBothAggregators(perc_users, labels, faulty, malicious)


def testBothAgg30ByzClientsMNISTExperiment():
    perc_users = torch.tensor([0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2,
                               0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2,
                               0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 10, 0.2, 0.2,
                               ])
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    faulty = [2, 10, 13]
    malicious = [15, 18]
    _testBothAggregators(perc_users, labels, faulty, malicious)

def _testAggregators(perc_users, labels, faulty, malicious):
    errorsDict = dict()
    for aggregator in aggregators:
        name = aggregator.__name__.replace("Aggregator", "")
        logPrint("TRAINING {}...".format(name))
        errorsDict[name] = trainOnMNIST(aggregator, perc_users, labels, faulty, malicious)

    plt.figure()
    i = 0
    colors = ['b', 'k', 'r', 'g']
    for name, err in errorsDict.items():
        plt.plot(err.numpy(), color=colors[i])
        i += 1
    plt.legend(errorsDict.keys())
    plt.show()


def _testPrivacyPreservingAggregators(perc_users, labels, faulty, malicious):
    errorsDict = dict()
    for aggregator in aggregators:
        name = aggregator.__name__.replace("Aggregator", "")
        logPrint("TRAINING PRIVACY PRESERVING {}...".format(name))
        errorsDict[name] = trainOnMNIST(aggregator, perc_users, labels, faulty, malicious, privacyPreserving=True)

    plt.figure()
    i = 0
    colors = ['b', 'k', 'r', 'g']
    for name, err in errorsDict.items():
        plt.plot(err.numpy(), color=colors[i])
        i += 1
    plt.legend(errorsDict.keys())
    plt.show()


def _testBothAggregators(perc_users, labels, faulty, malicious):
    errorsDict = dict()
    for aggregator in aggregators:
        name = aggregator.__name__.replace("Aggregator", "")

        logPrint("TRAINING VANILLA {}...".format(name))
        errorsDict[name] = trainOnMNIST(aggregator, perc_users, labels, faulty, malicious,
                                        privacyPreserving=False)

        name += " + DP"
        logPrint("TRAINING PRIVACY PRESERVING {}...".format(name))
        errorsDict[name] = trainOnMNIST(aggregator, perc_users, labels, faulty, malicious,
                                        privacyPreserving=True)

    plt.figure()
    i = 0
    colors = ['b', 'g', 'c', 'm', 'y', 'k', 'r']
    for name, err in errorsDict.items():
        plt.plot(err.numpy(), color=colors[i])
        i += 1
    plt.legend(errorsDict.keys())
    plt.show()


logPrint("Experiment started.")
# noByzClientMNISTExperiment()
# byzClientMNISTExperiment()
# privacyPreservingNoByzClientMNISTExperiment()
# privacyPreservingAndVanillaNoByzClientMNIST()

# privacyPreservingAndVanillaNoByzClientMNIST()
# privacyPreservingFewClientsMNISTExperiment()
# privacyPreserving30ClientsMNISTExperiment()
# noByzClientMNISTExperiment()
# noDP30ClientsMNISTExperiment()
# privacyPreserving30ClientsMNISTExperiment()
# testBothAgg30ClientsMNISTExperiment()
testBothAgg30ByzClientsMNISTExperiment()
# byzClientMNISTExperiment()

# privacyPreservingFewClientsMNISTExperiment()
