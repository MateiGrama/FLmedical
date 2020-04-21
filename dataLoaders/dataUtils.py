import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

from logger import logPrint


class DataLoader:
    """Abstract class used for specifying the data loading workflow """

    def loadData(self, percUsers, labels):
        raise Exception("LoadData method should be override by child class, "
                        "specific to the loaded dataset strategy.")

    @staticmethod
    def _filterByLabelAndShuffle(labels, xTrn, xTst, yTrn, yTst):
        xTrain = torch.tensor([])
        xTest = torch.tensor([])
        yTrain = torch.tensor([], dtype=torch.long)
        yTest = torch.tensor([], dtype=torch.long)
        # Extract the entries corresponding to the labels passed as param
        for e in labels:
            idx = (yTrn == e)
            xTrain = torch.cat((xTrain, xTrn[idx, :]), dim=0)
            yTrain = torch.cat((yTrain, yTrn[idx]), dim=0)

            idx = (yTst == e)
            xTest = torch.cat((xTest, xTst[idx, :]), dim=0)
            yTest = torch.cat((yTest, yTst[idx]), dim=0)
        # Shuffle
        r = torch.randperm(xTrain.size(0))
        xTrain = xTrain[r, :]
        yTrain = yTrain[r]
        return xTest, xTrain, yTest, yTrain


class DataLoaderMNIST(DataLoader):

    def loadData(self, percUsers, labels):
        logPrint("Loading MNIST...")
        percUsers = percUsers / percUsers.sum()
        userNo = percUsers.size(0)

        xTrn, yTrn, xTst, yTst = self.__loadMNISTdata()

        xTest, xTrain, yTest, yTrain = self._filterByLabelAndShuffle(labels, xTrn, xTst, yTrn, yTst)

        # Splitting data to users corresponding to user percentage param
        ntr_users = (percUsers * xTrain.size(0)).floor().numpy()

        training_data = []
        training_labels = []

        it = 0
        for i in range(userNo):
            x = xTrain[it:it + int(ntr_users[i]), :].clone().detach()
            y = yTrain[it:it + int(ntr_users[i])].clone().detach()
            training_data.append(x)
            training_labels.append(y)
            it = it + int(ntr_users[i])

        return training_data, training_labels, xTest, yTest

    @staticmethod
    def __loadMNISTdata():
        trans = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (1.0,))])

        # if not exist, download mnist dataset
        trainSet = dset.MNIST('data', train=True, transform=trans, download=True)
        testSet = dset.MNIST('data', train=False, transform=trans, download=True)

        # Scale pixel intensities to [-1, 1]
        x = trainSet.train_data
        x = 2 * (x.float() / 255.0) - 1
        # list of 2D images to 1D pixel intensities
        x = x.flatten(1, 2)
        y = trainSet.train_labels

        # Shuffle
        r = torch.randperm(x.size(0))
        xTrain = x[r]
        yTrain = y[r]

        # Scale pixel intensities to [-1, 1]
        xTest = testSet.test_data.clone().detach()
        xTest = 2 * (xTest.float() / 255.0) - 1
        # list of 2D images to 1D pixel intensities
        xTest = xTest.flatten(1, 2)
        yTest = testSet.test_labels.clone().detach()

        return xTrain, yTrain, xTest, yTest

        # WIP: postponed until decided to use or not PySyft

    def getSyftMNIST(self, percUsers, labels):
        percUsers = percUsers / percUsers.sum()
        userNo = percUsers.size(0)

        xTrn, yTrn, xTst, yTst = self.__loadMNISTdata()

        xTest, xTrain, yTest, yTrain = self.__filterByLabelAndShuffle(labels, xTrn, xTst, yTrn, yTst)

        # Splitting data to users corresponding to user percentage param
        ntr_users = (percUsers * xTrain.size(0)).floor().numpy()

        training_data = []
        training_labels = []

        it = 0
        for i in range(userNo):
            x = xTrain[it:it + int(ntr_users[i]), :].clone().detach()
            y = yTrain[it:it + int(ntr_users[i])].clone().detach()
            it = it + int(ntr_users[i])

            # create syft client sending them x, y

        return training_data, training_labels, xTest, yTest


class DataLoaderCOVID(DataLoader):
    pass
