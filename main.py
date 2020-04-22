from experiment.DefaultExperimentConfiguration import DefaultExperimentConfiguration
from dataLoaders.dataUtils import DataLoaderMNIST, DataLoaderCOVIDx
from classifiers import MNIST, CovidNet
from logger import logPrint
from client import Client
import aggregators as agg

import matplotlib.pyplot as plt
from itertools import product
import numpy as np
import random
import torch
import time


def __experimentOnMNIST(config):
    dataLoader = DataLoaderMNIST().loadData
    classifier = MNIST.Classifier
    __experimentSetup(config, dataLoader, classifier)


def __experimentOnCONVIDx(config):
    dataLoader = DataLoaderCOVIDx().loadData
    classifier = CovidNet  # .Classifier
    __experimentSetup(config, dataLoader, classifier)


def __experimentSetup(config, dataLoader, classifier):
    errorsDict = dict()
    for aggregator in config.aggregators:
        if config.privacyPreserve is not None:
            name = aggregator.__name__.replace("Aggregator", (" with DP" if config.privacyPreserve else ""))
            logPrint("TRAINING {}...".format(name))
            errorsDict[name] = __runExperiment(config, dataLoader, classifier,
                                               aggregator, config.privacyPreserve)
        else:
            name = aggregator.__name__.replace("Aggregator", "")
            logPrint("TRAINING {}...".format(name))
            errorsDict[name] = __runExperiment(config, dataLoader, classifier, aggregator,
                                               useDifferentialPrivacy=False)
            logPrint("TRAINING {} with DP...".format(name))
            errorsDict[name] = __runExperiment(config, dataLoader, classifier, aggregator,
                                               useDifferentialPrivacy=True)

    if config.plotResults:
        plt.figure()
        i = 0
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:cyan',
                  'tab:purple', 'tab:pink', 'tab:olive', 'tab:brown', 'tab:gray']
        for name, err in errorsDict.items():
            plt.plot(err.numpy(), color=colors[i])
            i += 1
        plt.legend(errorsDict.keys())
        plt.show()


def __runExperiment(config, dataLoader, classifier, aggregator, useDifferentialPrivacy):
    training_data, training_labels, xTest, yTest = dataLoader(config.percUsers, config.labels)
    clients = __initClients(config, training_data, training_labels)
    model = classifier().to(config.device)
    aggregator = aggregator(clients, model, config.rounds, config.device)
    return aggregator.trainAndTest(xTest, yTest)


def __initClients(config, training_data, training_labels):
    usersNo = config.percUsers.size(0)
    p0 = 1 / usersNo
    # Seed
    logPrint("Creating clients...")
    clients = []
    for i in range(usersNo):
        clients.append(Client(idx=i + 1,
                              x=training_data[i],
                              y=training_labels[i],
                              p=p0,
                              epochs=config.epochs,
                              batchSize=config.batchSize,
                              device=config.device,
                              useDifferentialPrivacy=config.privacyPreserve,
                              epsilon1=config.epsilon1,
                              epsilon3=config.epsilon3,
                              needClip=config.needClip,
                              clipValue=config.clipValue,
                              needNormalization=config.needNormalization,
                              releaseProportion=config.releaseProportion))

    ntr = 0
    for client in clients:
        ntr += client.xTrain.size(0)

    # Weight the value of the update of each user according to the number of training data points
    for client in clients:
        client.p = client.xTrain.size(0) / ntr
        # logPrint("Weight for user ", u.id, ": ", round(u.p,3))

    # Create malicious (byzantine) users
    for client in clients:
        if client.id in config.faulty:
            client.byz = True
            logPrint("User ", client.id, " is faulty.")
        if client.id in config.flipping:
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
def experiment(exp):
    def decorator():
        __setRandomSeeds(2)
        logPrint("Experiment {} began.".format(exp.__name__))
        begin = time.time()
        exp()
        end = time.time()
        logPrint("Experiment {} took {}".format(exp.__name__, end - begin))

    return decorator


@experiment
def noDP_noByzClient_onMNIST():
    configuration = DefaultExperimentConfiguration()

    __experimentOnMNIST(configuration)


@experiment
def withDP_withByzClient_onMNIST():
    configuration = DefaultExperimentConfiguration()

    configuration.percUsers = torch.tensor([0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1])
    configuration.labels = torch.tensor([0, 2, 5, 8])
    configuration.faulty = [2, 6]
    configuration.malicious = [1]

    __experimentOnMNIST(configuration)


@experiment
def withDP_noByzClient_onMNIST():
    configuration = DefaultExperimentConfiguration()

    configuration.labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    configuration.faulty = []
    configuration.malicious = []
    configuration.privacyPreserve = True

    __experimentOnMNIST(configuration)


@experiment
def withAndWithoutDP_noByzClient_onMNIST():
    configuration = DefaultExperimentConfiguration()

    configuration.privacyPreserve = None

    __experimentOnMNIST(configuration)


@experiment
def withDP_withByzClient_onMNIST():
    configuration = DefaultExperimentConfiguration()

    configuration.percUsers = torch.tensor([0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1])
    configuration.labels = torch.tensor([0, 2, 5, 8])
    configuration.faulty = [2, 6]
    configuration.malicious = [1]
    configuration.privacyPreserve = True

    __experimentOnMNIST(configuration)


@experiment
def withDP_fewNotByzClient_onMNIST():
    configuration = DefaultExperimentConfiguration()

    configuration.percUsers = torch.tensor([0.3, 0.25, 0.45])
    configuration.labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    configuration.privacyPreserve = True

    __experimentOnMNIST(configuration)


@experiment
def noDP_30notByzClients_onMNIST():
    configuration = DefaultExperimentConfiguration()

    configuration.percUsers = torch.tensor([0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2,
                                            0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2,
                                            0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2])

    __experimentOnMNIST(configuration)


@experiment
def withDP_30Clients_onMNIST():
    configuration = DefaultExperimentConfiguration()

    configuration.percUsers = torch.tensor([0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2,
                                            0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2,
                                            0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2])
    configuration.privacyPreserve = True

    __experimentOnMNIST(configuration)


@experiment
def withAndWithoutDP_30notByzClients_onMNIST():
    configuration = DefaultExperimentConfiguration()

    configuration.percUsers = torch.tensor([0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2,
                                            0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2,
                                            0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2])
    configuration.privacyPreserve = None

    __experimentOnMNIST(configuration)


@experiment
def withAndWithoutDP_30withByzClients_onMNIST():
    configuration = DefaultExperimentConfiguration()

    configuration.percUsers = torch.tensor([0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2,
                                            0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2,
                                            0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 10, 0.2, 0.2])
    configuration.faulty = [2, 10, 13]
    configuration.malicious = [15, 18]
    configuration.privacyPreserve = None

    __experimentOnMNIST(configuration)


@experiment
def noDP_noByzClient_fewRounds_onMNIST():
    configuration = DefaultExperimentConfiguration()
    configuration.rounds = 3
    configuration.plotResults = True
    __experimentOnMNIST(configuration)


@experiment
def withMultipleDPconfigsAndWithout_30notByzClients():
    releaseProportion = {0.1, 0.4}
    epsilon1 = {1, 0.01, 0.0001}
    epsilon3 = {1, 0.01, 0.0001}
    clipValues = {0.01, 0.0001}
    needClip = {False, True}
    needNormalise = {False, True}

    percUsers = torch.tensor([0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2,
                              0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2,
                              0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2])
    # Without DP
    noDPconfig = DefaultExperimentConfiguration()
    noDPconfig.aggregators = agg.allAggregators()
    noDPconfig.percUsers = percUsers
    __experimentOnMNIST(noDPconfig)

    # With DP
    for config in product(needClip, clipValues, epsilon1, epsilon3,
                          needNormalise, releaseProportion):
        needClip, clipValues, epsilon1, epsilon3, needNormalise, releaseProportion = config

        expConfig = DefaultExperimentConfiguration()
        expConfig.percUsers = percUsers
        expConfig.aggregators = agg.allAggregators()

        expConfig.privacyPreserve = True
        expConfig.releaseProportion = releaseProportion
        expConfig.needNormalise = needNormalise
        expConfig.clipValues = clipValues
        expConfig.needClip = needClip
        expConfig.epsilon1 = epsilon1
        expConfig.epsilon3 = epsilon3

        __experimentOnMNIST(expConfig)


@experiment
def noDP_noByzClient_onCOVIDx():
    configuration = DefaultExperimentConfiguration()

    __experimentOnCONVIDx(configuration)


noDP_noByzClient_onCOVIDx()
