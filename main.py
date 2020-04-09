import torch
import matplotlib.pyplot as plt

from aggregator import aggregators
import classifierMNIST
from client import Client
from dataUtils import getMNISTdata

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classifier = classifierMNIST.Classifier

# TRAINING PARAMETERS
rounds = 20  # TOTAL NUMBER OF TRAINING ROUNDS
epochs = 10  # NUMBER OF EPOCHS RUN IN EACH CLIENT BEFORE SENDING BACK THE MODEL UPDATE
batch_size = 200  # BATCH SIZE


def trainOnMNIST(aggregator, perc_users, labels, faulty, flipping):
    print("Loading MNIST...")
    training_data, training_labels, xTest, yTest = getMNISTdata(perc_users, labels)

    clients = initClients(perc_users, training_data, training_labels, faulty, flipping)

    # CREATE MODEL
    model = classifier().to(device)
    aggregator = aggregator(clients, model, rounds, device)
    return aggregator.trainAndTest(xTest, yTest)


def initClients(perc_users, training_data, training_labels, faulty, flipping):
    usersNo = perc_users.size(0)
    p0 = 1 / usersNo
    # Seed
    torch.manual_seed(2)
    print("Creating clients...")
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
        # print("Weight for user ", u.id, ": ", round(u.p,3))

    # Create malicious (byzantine) users
    for client in clients:
        if client.id in faulty:
            client.byz = True
            print("User ", client.id, " is faulty.")
        if client.id in flipping:
            client.flip = True
            print("User ", client.id, " is malicious.")
            # Flip labels
            # r = torch.randperm(u.ytr.size(0))
            # u.yTrain = u.yTrain[r]
            client.yTrain = torch.zeros(client.yTrain.size(), dtype=torch.int64)
    return clients


# perc_users = torch.tensor([0.2, 0.10, 0.15, 0.15, 0.15, 0.15, 0.1])
# labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# faulty = []
# malicious = []

perc_users = torch.tensor([0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1])
labels = torch.tensor([0, 2, 5, 8])
faulty = [2, 6]
malicious = [1]

errorsDict = dict()
for aggregator in aggregators:
    name = aggregator.__name__.replace("Aggregator", "")
    print("TRAINING {}...".format(name))
    errorsDict[name] = trainOnMNIST(aggregator, perc_users, labels, faulty, malicious)


plt.figure()
i = 0
colors = ['b', 'k', 'r', 'g']
for name, err in errorsDict.items():
    plt.plot(err.numpy(), color=colors[i])
    i += 1
plt.legend(errorsDict.keys())
plt.show()

exit(0)

