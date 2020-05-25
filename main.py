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
    datasetLoader = DatasetLoaderCOVIDx().getDatasets
    if model == 'COVIDNet':
        classifier = CovidNet.Classifier
    elif model == 'resnet18':
        classifier = CNN.Classifier
    else:
        raise Exception("Invalid Covid model name.")
    __experimentSetup(config, datasetLoader, classifier)


def __experimentSetup(config, datasetLoader, classifier):
    errorsDict = dict()
    for aggregator in config.aggregators:
        if config.privacyPreserve is not None:
            name = aggregator.__name__.replace("Aggregator", (" with DP" if config.privacyPreserve else ""))
            name += ":" + config.name if config.name else ""
            logPrint("TRAINING {}".format(name))
            errorsDict[name] = __runExperiment(config, datasetLoader, classifier,
                                               aggregator, config.privacyPreserve)
        else:
            name = aggregator.__name__.replace("Aggregator", "")
            name += ":" + config.name if config.name else ""
            logPrint("TRAINING {}".format(name))
            errorsDict[name] = __runExperiment(config, datasetLoader, classifier, aggregator,
                                               useDifferentialPrivacy=False)
            logPrint("TRAINING {} with DP".format(name))
            errorsDict[name] = __runExperiment(config, datasetLoader, classifier, aggregator,
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


def __runExperiment(config, datasetLoader, classifier, aggregator, useDifferentialPrivacy):
    trainDatasets, testDataset = datasetLoader(config.percUsers, config.labels, config.datasetSize)
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
                              epochs=config.epochs,
                              batchSize=config.batchSize,
                              learningRate=config.learningRate,
                              p=p0,
                              alpha=config.alpha,
                              beta=config.beta,
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
def withAndWithoutDP_AFA_30ByzAndNotClients_onMNIST():
    # Privacy budget = (releaseProportion, epsilon1, epsilon3)
    privacyBudget = [(0.1, 0.0001, 0.0001, 'low')]
    # Attacks: Malicious/Flipping - flips labels to 0; Faulty/Byzantine - noisy
    attacks = [([2 * i + 1 for i in range(2)], [], '2_faulty'),
               ([2 * i + 1 for i in range(4)], [], '4_faulty'),
               ([2 * i + 1 for i in range(6)], [], '6_faulty'),
               ([2 * i + 1 for i in range(8)], [], '8_faulty'),
               ([2 * i + 1 for i in range(9)], [], '9_faulty'),
               ([2 * i + 1 for i in range(10)], [], '10_faulty'),
               ([2 * i + 1 for i in range(12)], [], '12_faulty'),
               ([2 * i + 1 for i in range(14)], [], '14_faulty'),
               ([2 * i + 1 for i in range(15)], [], '15_faulty'),
               ([], [2 * i + 2 for i in range(2)], '2_malicious'),
               ([], [2 * i + 2 for i in range(4)], '4_malicious'),
               ([], [2 * i + 2 for i in range(6)], '6_malicious'),
               ([], [2 * i + 2 for i in range(8)], '8_malicious'),
               ([], [2 * i + 2 for i in range(9)], '9_malicious'),
               ([], [2 * i + 2 for i in range(10)], '10_malicious'),
               ([], [2 * i + 2 for i in range(12)], '12_malicious'),
               ([], [2 * i + 2 for i in range(14)], '14_malicious'),
               ([], [2 * i + 2 for i in range(15)], '15_malicious'),
               ([2 * i + 1 for i in range(1)], [2 * i + 2 for i in range(1)], '1_faulty,1_malicious'),
               ([2 * i + 1 for i in range(2)], [2 * i + 2 for i in range(2)], '2_faulty,2_malicious'),
               ([2 * i + 1 for i in range(3)], [2 * i + 2 for i in range(3)], '3_faulty,3_malicious'),
               ([2 * i + 1 for i in range(4)], [2 * i + 2 for i in range(4)], '4_faulty,4_malicious'),
               ([2 * i + 1 for i in range(5)], [2 * i + 2 for i in range(5)], '5_faulty,5_malicious'),
               ([2 * i + 1 for i in range(6)], [2 * i + 2 for i in range(6)], '6_faulty,6_malicious'),
               ([2 * i + 1 for i in range(7)], [2 * i + 2 for i in range(7)], '7_faulty,7_malicious'),
               ([2 * i + 1 for i in range(8)], [2 * i + 2 for i in range(8)], '8_faulty,8_malicious')]

    attacks = [([2 * i + 1 for i in range(7)], [], '7_faulty'),
               ([], [2 * i + 2 for i in range(7)], '7_malicious')]

    percUsers = torch.tensor([0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2,
                              0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2,
                              0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2])

    # Without DP without attacks
    noDPconfig = DefaultExperimentConfiguration()
    noDPconfig.aggregators = [agg.AFAAggregator]
    noDPconfig.percUsers = percUsers
    __experimentOnMNIST(noDPconfig)

    # Without DP
    for scenario in attacks:
        faulty, malicious, attackName = scenario
        noDPconfig = DefaultExperimentConfiguration()
        noDPconfig.aggregators = [agg.AFAAggregator]
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
        expConfig.aggregators = [agg.AFAAggregator]

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
def withAndWithoutDP_manyAlphaBetaAFA_30ByzAndNotClients_onMNIST():
    # Privacy budget = (releaseProportion, epsilon1, epsilon3)
    privacyBudget = [(0.1, 0.0001, 0.0001, 'low')]
    # Attacks: Malicious/Flipping - flips labels to 0; Faulty/Byzantine - noisy
    attacks = [
        ([2 * i + 1 for i in range(2)], [], '2_faulty'),
        # ([2 * i + 1 for i in range(4)], [], '4_faulty'),
        ([2 * i + 1 for i in range(6)], [], '6_faulty'),
        # ([2 * i + 1 for i in range(7)], [], '7_faulty'),
        ([2 * i + 1 for i in range(8)], [], '8_faulty'),
        # ([2 * i + 1 for i in range(9)], [], '9_faulty'),
        ([2 * i + 1 for i in range(10)], [], '10_faulty'),
        # ([2 * i + 1 for i in range(12)], [], '12_faulty'),
        # ([2 * i + 1 for i in range(14)], [], '14_faulty'),
        # ([2 * i + 1 for i in range(15)], [], '15_faulty'),
        ([], [2 * i + 2 for i in range(2)], '2_malicious'),
        # ([], [2 * i + 2 for i in range(4)], '4_malicious'),
        ([], [2 * i + 2 for i in range(6)], '6_malicious'),
        # ([], [2 * i + 2 for i in range(7)], '7_malicious'),
        ([], [2 * i + 2 for i in range(8)], '8_malicious'),
        # ([], [2 * i + 2 for i in range(9)], '9_malicious'),
        ([], [2 * i + 2 for i in range(10)], '10_malicious'),
        # ([], [2 * i + 2 for i in range(12)], '12_malicious'),
        # ([], [2 * i + 2 for i in range(14)], '14_malicious'),
        # ([], [2 * i + 2 for i in range(15)], '15_malicious'),
        ([2 * i + 1 for i in range(1)], [2 * i + 2 for i in range(1)], '1_faulty,1_malicious'),
        # ([2 * i + 1 for i in range(2)], [2 * i + 2 for i in range(2)], '2_faulty,2_malicious'),
        # ([2 * i + 1 for i in range(3)], [2 * i + 2 for i in range(3)], '3_faulty,3_malicious'),
        ([2 * i + 1 for i in range(4)], [2 * i + 2 for i in range(4)], '4_faulty,4_malicious'),
        # ([2 * i + 1 for i in range(5)], [2 * i + 2 for i in range(5)], '5_faulty,5_malicious'),
        # ([2 * i + 1 for i in range(6)], [2 * i + 2 for i in range(6)], '6_faulty,6_malicious'),
        # ([2 * i + 1 for i in range(7)], [2 * i + 2 for i in range(7)], '7_faulty,7_malicious'),
        # ([2 * i + 1 for i in range(8)], [2 * i + 2 for i in range(8)], '8_faulty,8_malicious')
    ]

    # Workaround to run experiments in parallel runs:
    e = 3  # experiment index
    nAttacks = 2  # number of attack scenarios considered per experiement
    attacks = attacks[e * nAttacks: e * nAttacks + nAttacks]

    alphaBetas = [(2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3), (4, 4)]

    percUsers = torch.tensor([0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2,
                              0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2,
                              0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2])

    # # Only run the vanilla experiment once
    # if not e:
    #     # Without DP without attacks
    #     for alphaBeta in alphaBetas:
    #         alpha, beta = alphaBeta
    #
    #         noDPconfig = DefaultExperimentConfiguration()
    #         noDPconfig.aggregators = [agg.AFAAggregator]
    #         noDPconfig.percUsers = percUsers
    #
    #         noDPconfig.alpha = alpha
    #         noDPconfig.beta = beta
    #
    #         noDPconfig.name = "alphaBeta:{};".format(alphaBeta)
    #
    #         __experimentOnMNIST(noDPconfig)
    #
    alphaBetas = [(3, 4), (4, 3), (4, 4)]
    # Without DP
    for alphaBeta, attack in product(alphaBetas, attacks):
        faulty, malicious, attackName = attack
        alpha, beta = alphaBeta
        noDPconfig = DefaultExperimentConfiguration()
        noDPconfig.aggregators = [agg.AFAAggregator]
        noDPconfig.percUsers = percUsers

        noDPconfig.faulty = faulty
        noDPconfig.malicious = malicious
        noDPconfig.name = "alphaBeta:{};".format(alphaBeta)
        noDPconfig.name += "altered:{};".format(attackName)

        __experimentOnMNIST(noDPconfig)

    alphaBetas = [(2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3), (4, 4)]
    # With DP
    for budget, alphaBeta, attack in product(privacyBudget, alphaBetas, attacks):
        releaseProportion, epsilon1, epsilon3, budgetName = budget
        faulty, malicious, attackName = attack
        alpha, beta = alphaBeta

        expConfig = DefaultExperimentConfiguration()
        expConfig.percUsers = percUsers
        expConfig.aggregators = [agg.AFAAggregator]

        expConfig.privacyPreserve = True
        expConfig.releaseProportion = releaseProportion
        expConfig.epsilon1 = epsilon1
        expConfig.epsilon3 = epsilon3
        expConfig.needClip = True

        expConfig.alpha = alpha
        expConfig.beta = beta

        expConfig.faulty = faulty
        expConfig.malicious = malicious

        expConfig.name = "alphaBeta:{};".format(alphaBeta)
        expConfig.name += "altered:{};".format(attackName)
        expConfig.name += "privacyBudget:{};".format(budgetName)

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
def withAndWithoutDP_withAndWithoutByz_10ByzClients_onCOVIDx():
    epsilon1 = 0.0001
    epsilon3 = 0.0001
    releaseProportion = 0.1

    learningRate = 0.00002
    batchSize = 2
    rounds = 25

    percUsers = torch.tensor([0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2])

    # Without DP without attacks
    noDPconfig = DefaultExperimentConfiguration()
    noDPconfig.aggregators = agg.allAggregators()
    noDPconfig.learningRate = learningRate
    noDPconfig.batchSize = batchSize
    noDPconfig.rounds = rounds

    noDPconfig.percUsers = percUsers

    __experimentOnCONVIDx(noDPconfig)

    # With DP without attacks
    DPconfig = DefaultExperimentConfiguration()
    DPconfig.aggregators = agg.allAggregators()
    DPconfig.learningRate = learningRate
    DPconfig.batchSize = batchSize
    DPconfig.rounds = rounds

    DPconfig.privacyPreserve = True
    DPconfig.releaseProportion = releaseProportion
    DPconfig.epsilon1 = epsilon1
    DPconfig.epsilon3 = epsilon3
    DPconfig.needClip = True

    noDPconfig.percUsers = percUsers

    __experimentOnCONVIDx(DPconfig)

    # With DP with one attacker
    DPconfig = DefaultExperimentConfiguration()
    DPconfig.aggregators = agg.allAggregators()
    DPconfig.learningRate = learningRate
    DPconfig.batchSize = batchSize
    DPconfig.rounds = rounds

    DPconfig.privacyPreserve = True
    DPconfig.releaseProportion = releaseProportion
    DPconfig.epsilon1 = epsilon1
    DPconfig.epsilon3 = epsilon3
    DPconfig.needClip = True

    noDPconfig.percUsers = percUsers

    DPconfig.malicious = [3]
    DPconfig.name = "altered:1_malicious"

    __experimentOnCONVIDx(DPconfig)

    # With DP with more attackers
    DPbyzConfig = DefaultExperimentConfiguration()
    DPbyzConfig.aggregators = agg.allAggregators()
    DPbyzConfig.learningRate = learningRate
    DPbyzConfig.batchSize = batchSize
    DPbyzConfig.rounds = rounds

    DPbyzConfig.privacyPreserve = True
    DPbyzConfig.releaseProportion = releaseProportion
    DPbyzConfig.epsilon1 = epsilon1
    DPbyzConfig.epsilon3 = epsilon3
    DPbyzConfig.needClip = True

    noDPconfig.percUsers = percUsers

    DPbyzConfig.faulty = [1]
    DPbyzConfig.malicious = [2, 4]

    DPbyzConfig.name = "altered:1_faulty,2_malicious"

    __experimentOnCONVIDx(DPbyzConfig)


@experiment
def withAndWithoutDP_withAndWithoutByz_5ByzClients_resnet_onCOVIDx():
    epsilon1 = 0.0001
    epsilon3 = 0.0001
    releaseProportion = 0.1

    learningRate = 0.00002
    batchSize = 2
    rounds = 10

    percUsers = torch.tensor([0.1, 0.15, 0.2, 0.2, 0.1])

    # Without DP without attacks
    noDPconfig = DefaultExperimentConfiguration()
    noDPconfig.aggregators = agg.allAggregators()
    noDPconfig.learningRate = learningRate
    noDPconfig.batchSize = batchSize
    noDPconfig.rounds = rounds

    noDPconfig.percUsers = percUsers

    __experimentOnCONVIDx(noDPconfig, model='resnet18')

    # With DP without attacks
    DPconfig = DefaultExperimentConfiguration()
    DPconfig.aggregators = agg.allAggregators()
    DPconfig.learningRate = learningRate
    DPconfig.batchSize = batchSize
    DPconfig.rounds = rounds

    DPconfig.privacyPreserve = True
    DPconfig.releaseProportion = releaseProportion
    DPconfig.epsilon1 = epsilon1
    DPconfig.epsilon3 = epsilon3
    DPconfig.needClip = True

    noDPconfig.percUsers = percUsers

    __experimentOnCONVIDx(DPconfig, model='resnet18')

    # With DP with one attacker
    DPconfig = DefaultExperimentConfiguration()
    DPconfig.aggregators = agg.allAggregators()
    DPconfig.learningRate = learningRate
    DPconfig.batchSize = batchSize
    DPconfig.rounds = rounds

    DPconfig.privacyPreserve = True
    DPconfig.releaseProportion = releaseProportion
    DPconfig.epsilon1 = epsilon1
    DPconfig.epsilon3 = epsilon3
    DPconfig.needClip = True

    noDPconfig.percUsers = percUsers

    DPconfig.malicious = [3]
    DPconfig.name = "altered:1_malicious"

    __experimentOnCONVIDx(DPconfig, model='resnet18')

    # With DP with more attackers
    DPbyzConfig = DefaultExperimentConfiguration()
    DPbyzConfig.aggregators = agg.allAggregators()
    DPbyzConfig.learningRate = learningRate
    DPbyzConfig.batchSize = batchSize
    DPbyzConfig.rounds = rounds

    DPbyzConfig.privacyPreserve = True
    DPbyzConfig.releaseProportion = releaseProportion
    DPbyzConfig.epsilon1 = epsilon1
    DPbyzConfig.epsilon3 = epsilon3
    DPbyzConfig.needClip = True

    noDPconfig.percUsers = percUsers

    DPbyzConfig.faulty = [3]
    DPbyzConfig.malicious = [5]

    DPbyzConfig.name = "altered:1_faulty,1_malicious"

    __experimentOnCONVIDx(DPbyzConfig, model='resnet18')


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

withAndWithoutDP_manyAlphaBetaAFA_30ByzAndNotClients_onMNIST()
