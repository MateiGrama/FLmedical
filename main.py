from experiment.DefaultExperimentConfiguration import DefaultExperimentConfiguration
from datasetLoaders.loaders import DatasetLoaderMNIST, DatasetLoaderCOVIDx
from classifiers import MNIST, CovidNet, CNN
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
    dataLoader = DatasetLoaderMNIST().getDatasets
    classifier = MNIST.Classifier
    __experimentSetup(config, dataLoader, classifier)


def __experimentOnCONVIDx(config, model='COVIDNet'):
    dataLoader = DatasetLoaderCOVIDx().getDatasets
    if model == 'COVIDNet':
        classifier = CovidNet.Classifier
    elif model == 'resnet18':
        classifier = CNN.Classifier
    else:
        raise Exception("Invalid Covid model name.")
    __experimentSetup(config, dataLoader, classifier)


def __experimentSetup(config, dataLoader, classifier):
    errorsDict = dict()
    for aggregator in config.aggregators:
        if config.privacyPreserve is not None:
            name = aggregator.__name__.replace("Aggregator", (" with DP" if config.privacyPreserve else ""))
            name += ":" + config.name if config.name else ""
            logPrint("TRAINING {}".format(name))
            errorsDict[name] = __runExperiment(config, dataLoader, classifier,
                                               aggregator, config.privacyPreserve)
        else:
            name = aggregator.__name__.replace("Aggregator", "")
            name += ":" + config.name if config.name else ""
            logPrint("TRAINING {}".format(name))
            errorsDict[name] = __runExperiment(config, dataLoader, classifier, aggregator,
                                               useDifferentialPrivacy=False)
            logPrint("TRAINING {} with DP".format(name))
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
    trainDatasets, testDataset = dataLoader(config.percUsers, config.labels, config.datasetSize)
    clients = __initClients(config, trainDatasets, useDifferentialPrivacy)
    model = classifier().to(config.device)
    aggregator = aggregator(clients, model, config.rounds, config.device)
    return aggregator.trainAndTest(testDataset)


def __initClients(config, trainDatasets, useDifferentialPrivacy):
    usersNo = config.percUsers.size(0)
    p0 = 1 / usersNo
    # Seed
    logPrint("Creating clients...")
    clients = []
    for i in range(usersNo):
        clients.append(Client(idx=i + 1,
                              trainDataset=trainDatasets[i],
                              p=p0,
                              epochs=config.epochs,
                              batchSize=config.batchSize,
                              learningRate=config.learningRate,
                              device=config.device,
                              useDifferentialPrivacy=useDifferentialPrivacy,
                              epsilon1=config.epsilon1,
                              epsilon3=config.epsilon3,
                              needClip=config.needClip,
                              clipValue=config.clipValue,
                              needNormalization=config.needNormalization,
                              releaseProportion=config.releaseProportion))

    nTrain = sum([client.n for client in clients])
    # Weight the value of the update of each user according to the number of training data points
    for client in clients:
        client.p = client.n / nTrain
        # logPrint("Weight for user ", u.id, ": ", round(u.p,3))

    # Create malicious (byzantine) and faulty users
    for client in clients:
        if client.id in config.faulty:
            client.byz = True
            logPrint("User", client.id, "is faulty.")
        if client.id in config.malicious:
            client.flip = True
            logPrint("User", client.id, "is malicious.")
            # Flip labels
            # r = torch.randperm(u.ytr.size(0))
            # u.yTrain = u.yTrain[r]
            client.trainDataset.zeroLabels()
    return clients


def __setRandomSeeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


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
    configuration.rounds = 7
    configuration.plotResults = True
    __experimentOnMNIST(configuration)


@experiment
def noDP_noByzClient_fewRounds_onMNIST():
    configuration = DefaultExperimentConfiguration()
    configuration.rounds = 3
    configuration.plotResults = True
    __experimentOnMNIST(configuration)


@experiment
def withMultipleDPconfigsAndWithout_30notByzClients_onMNIST():
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
def withMultipleDPandByzConfigsAndWithout_30ByzClients_onMNIST():
    # Privacy budget = (releaseProportion, epsilon1, epsilon3)
    privacyBudget = [(0.1, 0.0001, 0.0001, 'low'), (0.4, 1, 1, 'high')]

    # Attacks: Malicious/Flipping - flips labels to 0; Faulty/Byzantine - noisy
    attacks = [([1], [], '1_faulty'),
               ([], [2], '1_malicious'),
               ([1], [2], '1_faulty,1_malicious'),
               ([1, 3, 5, 7, 9], [2, 4, 6, 8, 10], '5_faulty,5_malicious'),
               ([1, 3, 5, 7, 9, 11, 13, 15, 17, 19], [], '10_faulty'),
               ([], [2, 4, 6, 8, 10, 12, 14, 16, 18, 20], '10_malicious'),
               ([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29],
                [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28], '15_faulty,14_malicious')]

    percUsers = torch.tensor([0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2,
                              0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2,
                              0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2])

    # Without DP without attacks
    noDPconfig = DefaultExperimentConfiguration()
    noDPconfig.aggregators = agg.allAggregators()
    noDPconfig.percUsers = percUsers
    __experimentOnMNIST(noDPconfig)

    # Without DP
    for scenario in attacks:
        faulty, malicious, attackName = scenario
        noDPconfig = DefaultExperimentConfiguration()
        noDPconfig.aggregators = agg.allAggregators()
        noDPconfig.percUsers = percUsers

        noDPconfig.faulty = faulty
        noDPconfig.malicious = malicious
        noDPconfig.name = "altered:{}".format(attackName)

        __experimentOnMNIST(noDPconfig)

    # With DP
    for budget, attack in product(privacyBudget, attacks):
        releaseProportion, epsilon1, epsilon3, budgetName = budget
        faulty, malicious, attackName = attack

        expConfig = DefaultExperimentConfiguration()
        expConfig.percUsers = percUsers
        expConfig.aggregators = agg.allAggregators()

        expConfig.privacyPreserve = True
        expConfig.releaseProportion = releaseProportion
        expConfig.epsilon1 = epsilon1
        expConfig.epsilon3 = epsilon3
        expConfig.needClip = True

        expConfig.faulty = faulty
        expConfig.malicious = malicious

        expConfig.name = "altered:{};".format(attackName)
        expConfig.name += "privacyBudget:{};".format(budgetName)

        __experimentOnMNIST(expConfig)


@experiment
def withLowAndHighAndWithoutDP_30ByzClients_onMNIST():
    # Privacy budget = (releaseProportion, epsilon1, epsilon3)
    privacyBudget = [(0.1, 0.0001, 0.0001, 'low'), (0.4, 1, 1, 'high')]

    # Attacks: Malicious/Flipping - flips labels to 0; Faulty/Byzantine - noisy
    attacks = [([1, 3, 5, 7, 9, 11, 13], [2, 4, 6, 8, 10, 12, 14], '7_faulty,7_malicious')]

    percUsers = torch.tensor([0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2,
                              0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2,
                              0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2])

    # Without DP
    for scenario in attacks:
        faulty, malicious, attackName = scenario
        noDPconfig = DefaultExperimentConfiguration()
        noDPconfig.aggregators = agg.allAggregators()
        noDPconfig.percUsers = percUsers

        noDPconfig.faulty = faulty
        noDPconfig.malicious = malicious
        noDPconfig.name = "altered:{}".format(attackName)

        __experimentOnMNIST(noDPconfig)

    # With DP
    for budget, attack in product(privacyBudget, attacks):
        releaseProportion, epsilon1, epsilon3, budgetName = budget
        faulty, malicious, attackName = attack

        expConfig = DefaultExperimentConfiguration()
        expConfig.percUsers = percUsers
        expConfig.aggregators = agg.allAggregators()

        expConfig.privacyPreserve = True
        expConfig.releaseProportion = releaseProportion
        expConfig.epsilon1 = epsilon1
        expConfig.epsilon3 = epsilon3
        expConfig.needClip = True

        expConfig.faulty = faulty
        expConfig.malicious = malicious

        expConfig.name = "altered:{};".format(attackName)
        expConfig.name += "privacyBudget:{};".format(budgetName)

        __experimentOnMNIST(expConfig)


@experiment
def withAndWithoutDP_withAndWithoutByz_30ByzClients_onCOVIDx():
    # Privacy budget = (releaseProportion, epsilon1, epsilon3)

    percUsers = torch.tensor([0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2,
                              0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2,
                              0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2])

    faulty = [5, 10]
    malicious = [3, 13, 23]

    epsilon1 = 0.0001
    epsilon3 = 0.0001
    releaseProportion = 0.1

    learningRate = 0.00002
    batchSize = 2

    # Without DP without attacks
    noDPconfig = DefaultExperimentConfiguration()
    noDPconfig.aggregators = agg.allAggregators()
    noDPconfig.percUsers = percUsers
    noDPconfig.learningRate = learningRate
    noDPconfig.batchSize = batchSize

    __experimentOnCONVIDx(noDPconfig)

    # With DP without attacks
    DPconfig = DefaultExperimentConfiguration()
    DPconfig.aggregators = agg.allAggregators()
    DPconfig.percUsers = percUsers
    DPconfig.learningRate = learningRate
    DPconfig.batchSize = batchSize

    DPconfig.privacyPreserve = True
    DPconfig.releaseProportion = releaseProportion
    DPconfig.epsilon1 = epsilon1
    DPconfig.epsilon3 = epsilon3
    DPconfig.needClip = True

    __experimentOnCONVIDx(DPconfig)

    # With DP with attacks
    DPbyzConfig = DefaultExperimentConfiguration()
    DPbyzConfig.percUsers = percUsers
    DPbyzConfig.aggregators = agg.allAggregators()
    DPbyzConfig.learningRate = learningRate
    DPbyzConfig.batchSize = batchSize

    DPbyzConfig.privacyPreserve = True
    DPbyzConfig.releaseProportion = releaseProportion
    DPbyzConfig.epsilon1 = epsilon1
    DPbyzConfig.epsilon3 = epsilon3
    DPbyzConfig.needClip = True

    DPbyzConfig.faulty = faulty
    DPbyzConfig.malicious = malicious

    DPbyzConfig.name = "altered:2_faulty,3_malicious"

    __experimentOnCONVIDx(DPbyzConfig)


@experiment
def noDP_noByzClient_onCOVIDx():
    configuration = DefaultExperimentConfiguration()
    configuration.batchSize = 64
    configuration.learningRate = 0.0002
    __experimentOnCONVIDx(configuration)


@experiment
def noDP_singleClient_onCOVIDx_100train11test():
    configuration = DefaultExperimentConfiguration()
    configuration.percUsers = torch.tensor([1., 2.])
    configuration.datasetSize = (100, 11)
    configuration.batchSize = 20
    configuration.epochs = 3
    configuration.learningRate = 0.0002
    __experimentOnCONVIDx(configuration)


@experiment
def customExperiment():
    configuration = DefaultExperimentConfiguration()

    configuration.percUsers = torch.tensor([0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2,
                                            0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2,
                                            0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2])

    configuration.malicious = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

    configuration.aggregators = [agg.AFAAggregator]

    __experimentOnMNIST(configuration)


# withMultipleDPandByzConfigsAndWithout_30ByzClients_onMNIST()

withAndWithoutDP_withAndWithoutByz_30ByzClients_onCOVIDx()
