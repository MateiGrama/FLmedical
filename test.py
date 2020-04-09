import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import syft as sy
from typing import List


class a:
    def b(self):
        return 1


c = [a]
d = c[0]()
# print("print:{}".format(d.b()))

a = dict()
a['a'] = 1
a['b'] = 2
a['c'] = 3

print(tuple(a.keys()))

exit(0)

workers_num = 3
hook = sy.TorchHook(torch)
workers = [sy.VirtualWorker(hook, id="worker" + str(i)) for i in range(workers_num)]


class Arguments():
    def __init__(self):
        self.batch_size = 1024
        self.test_batch_size = 1024
        self.epochs = 5
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 20
        self.save_model = False


args = Arguments()
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

mean, std = (0.5,), (0.5,)

# Create a transform and normalise data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean, std)])

# Download FMNIST training dataset and load training data
train_set = datasets.FashionMNIST('~/.pytorch/FMNIST/', download=True, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

# Download FMNIST test dataset and load test data
test_set = datasets.FashionMNIST('~/.pytorch/FMNIST/', download=True, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

###### Not using Pysyft FederatedDataLoader in order to implemet own Aggreagtion

# federated_train_loader = sy.FederatedDataLoader(
#     datasets.FashionMNIST('../data', train=True, download=True, transform=transform)
#             .federate((worker1, worker2, worker3)),
#     batch_size=args.batch_size, shuffle=True, **kwargs)

# test_loader = torch.utils.data.DataLoader(
#     datasets.FashionMNIST('../data', train=False, transform=transform),
#     batch_size=args.test_batch_size, shuffle=True, **kwargs)

remote_dataset = list(map(lambda _: list(), workers))

for batch_idx, (data, target) in enumerate(train_loader):
    data = data.send(workers[batch_idx % len(workers)])
    target = target.send(workers[batch_idx % len(workers)])

    remote_dataset[batch_idx % len(workers)].append((data, target))


class FMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)

        return x


# Aggregator model
model = FMNIST()

# worker models
models = dict([(worker, FMNIST())
               for worker in workers])
optimizers = dict([(model, optim.SGD(model.parameters(), lr=args.lr))
                   for model in models.values()])


def update(data, target, model, optimizer):
    model.send(data.location)
    prediction = model(data)
    loss = F.mse_loss(prediction.view(-1), target)
    loss.backward()
    optimizer.step()
    return model


def train(aggregator):
    for worker in workers:
        for data_index in range(len(remote_dataset[0]) - 1):
            worker_index = workers.index(worker)
            data, target = remote_dataset[worker_index][data_index]
            models[worker] = update(data, target, models[worker], optimizers[models[worker]])
    for model in models.values():
        model.get()
    # Results of the federated round to be aggregated
    results_dict = dict(map(lambda w: ("worker" + str(workers.index(w)), models[w]), workers))
    return aggregator(results_dict)


def test(model):
    model.eval()
    test_loss = 0
    for data, target in test_loader:
        output = model(data)
        test_loss += F.mse_loss(output.view(-1), target, reduction='sum').item()
        predection = output.data.max(1, keepdim=True)[1]

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}'.format(test_loss))


def add_model(dst_model, src_model):
    """Add the parameters of two models.
        Args:
            dst_model (torch.nn.Module): the model to which the src_model will be added.
            src_model (torch.nn.Module): the model to be added to dst_model.
        Returns:
            torch.nn.Module: the resulting model of the addition.
        """

    params1 = src_model.named_parameters()
    params2 = dst_model.named_parameters()
    dict_params2 = dict(params2)
    with torch.no_grad():
        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].set_(param1.data + dict_params2[name1].data)
    return dst_model


def scale_model(model, scale):
    """Scale the parameters of a model.
    Args:
        model (torch.nn.Module): the models whose parameters will be scaled.
        scale (float): the scaling factor.
    Returns:
        torch.nn.Module: the module with scaled parameters.
    """
    params = model.named_parameters()
    dict_params = dict(params)
    with torch.no_grad():
        for name, param in dict_params.items():
            dict_params[name].set_(dict_params[name].data * scale)
    return model


def federated_avg(models: List[torch.nn.Module]) -> torch.nn.Module:
    nr_models = len(models)
    model_list = list(models.values())
    model = model_list[0]
    for i in range(1, nr_models):
        model = add_model(model, model_list[i])
    model = scale_model(model, 1.0 / nr_models)
    return model


aggregator = federated_avg

for epoch in range(args.epochs):
    print(f"Epoch Number {epoch + 1}")
    model = train(aggregator)

    test(model)
    # print('Communication time over the network', round(total_time, 2), 's\n')
