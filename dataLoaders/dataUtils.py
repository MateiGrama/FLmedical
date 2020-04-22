import os
import sys
from shutil import copyfile

import cv2
import numpy as np
import pandas as pd
import pydicom as dicom
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, datasets

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
        return xTrain, yTrain, xTest, yTest

    @staticmethod
    def _splitTrainData(percUsers, xTest, xTrain, yTest, yTrain):
        percUsers = percUsers / percUsers.sum()
        userNo = percUsers.size(0)

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

    def _readFilePaths(file):
        paths, labels = [], []
        with open(file, 'r') as f:
            lines = f.read().splitlines()
            for idx, line in enumerate(lines):
                if '/ c o' in line:
                    break
                subjid, path, label = line.split(' ')
                paths.append(path)
                labels.append(label)
        return paths, labels


class DataLoaderMNIST(DataLoader):

    def loadData(self, percUsers, labels):
        logPrint("Loading MNIST...")
        data = self.__loadMNISTdata()
        xTrain, yTrain, xTest, yTest = self._filterByLabelAndShuffle(labels, *data)
        # Splitting data to users corresponding to user percentage param
        return self._splitTrainData(percUsers, xTest, xTrain, yTest, yTrain)

    @staticmethod
    def __loadMNISTdata():
        trans = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (1.0,))])

        # if not exist, download mnist dataset
        trainSet = datasets.MNIST('data', train=True, transform=trans, download=True)
        testSet = datasets.MNIST('data', train=False, transform=trans, download=True)

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

    # WIP: postponed until pysyft integration needed
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


class DataLoaderCOVIDx(DataLoader):

    def __init__(self, dim=(224, 224), assembleDatasets=True):
        self.assembleDatasets = assembleDatasets
        self.dim = dim

        self.COVIDxLabelsDict = {'pneumonia': 0, 'normal': 1, 'COVID-19': 2}

        self.dataPath = './data/COVIDx'
        self.testFile = self.dataPath + '/test_split_v2.txt'
        self.trainFile = self.dataPath + '/train_split_v2.txt'

    def loadData(self, percUsers, labels):
        logPrint("Loading COVIDx...")
        data = self.__loadCOVIDxData()
        xTrain, yTrain, xTest, yTest = self._filterByLabelAndShuffle(labels, *data)
        # Splitting data to users corresponding to user percentage param
        return self._splitTrainData(percUsers, xTest, xTrain, yTest, yTrain)

    def __loadCOVIDxData(self):
        if self.__datasetNotFound():
            logPrint("Can't find train|test split .txt files or "
                     "/train, /test files not populated accordingly.")
            if not self.assembleDatasets:
                sys.exit(0)

            logPrint("Proceeding to assemble dataset from downloaded resources.")
            self.__joinDatasets()

        trainPaths, trainLabels = self.__readFilePaths(self.trainFile)
        testPaths, testLabels = self.__readFilePaths(self.testFile)
        trainSize = len(trainPaths)
        testSize = len(testPaths)

        xTrain = torch.tensor([])
        xTest = torch.tensor([])
        yTrain = torch.tensor([], dtype=torch.long)
        yTest = torch.tensor([], dtype=torch.long)

        for i in range(trainSize):
            imageTensor = self.__load_image(self.dataPath + '/train/' + trainPaths[i], self.dim)
            xTrain = torch.cat((xTrain, torch.unsqueeze(imageTensor, dim=0)), dim=0)
            labelTensor = torch.tensor(self.COVIDxLabelsDict[trainLabels[i]])
            yTrain = torch.cat((yTrain, torch.unsqueeze(labelTensor, dim=0)), dim=0)
        for i in range(testSize):
            imageTensor = self.__load_image(self.dataPath + '/test/' + testPaths[i], self.dim)
            xTest = torch.cat((xTest, torch.unsqueeze(imageTensor, dim=0)), dim=0)
            labelTensor = torch.tensor(self.COVIDxLabelsDict[testLabels[i]])
            yTest = torch.cat((yTest, torch.unsqueeze(labelTensor, dim=0)), dim=0)

        return xTrain, yTrain, xTest, yTest

    def __datasetNotFound(self):
        if not os.path.exists(self.dataPath + "/test_split_v2.txt") or \
                not os.path.exists(self.dataPath + "/train_split_v2.txt") or \
                not os.path.exists(self.dataPath + "/test") or \
                not os.path.exists(self.dataPath + "/train") or \
                not len(os.listdir(self.dataPath + "/test")) or \
                not len(os.listdir(self.dataPath + "/train")):
            # Might also want to check that files count of
            # /test, /train folder match .txt files
            return True
        return False

    @staticmethod
    def __readFilePaths(filePath):
        paths, labels = [], []
        with open(filePath, 'r') as file:
            for line in file:
                _, path, label = line.replace('\n', '').split(' ')
                paths.append(path)
                labels.append(label)
            return paths, labels

    @staticmethod
    def __load_image(img_path, dim):
        if not os.path.exists(img_path):
            print("IMAGE DOES NOT EXIST {}".format(img_path))
        image = Image.open(img_path).convert('RGB')
        image = image.resize(dim).convert('RGB')

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
        # if(image_tensor.size(0)>1):
        #     #print(img_path," > 1 channels")
        #     image_tensor = image_tensor.mean(dim=0,keepdim=True)
        return transform(image)

    def __joinDatasets(self):
        dataSources = ['/covid-chestxray-dataset',
                       '/rsna-kaggle-dataset',
                       '/Figure1-covid-chestxray-dataset']
        if not len(os.listdir(self.dataPath + dataSources[0])):
            logPrint("You need to clone https://github.com/ieee8023/covid-chestxray-dataset to {}."
                     "".format(self.dataPath + dataSources[0]))
            exit(0)
        if not len(os.listdir(self.dataPath + dataSources[1])):
            logPrint("You need to unzip (https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) dataset to {}."
                     "".format(self.dataPath + dataSources[1]))
            exit(0)

        COPY_FILE = True
        if COPY_FILE:
            if not os.path.exists(self.dataPath + '/train'):
                os.makedirs(self.dataPath + '/train')
            if not os.path.exists(self.dataPath + '/test'):
                os.makedirs(self.dataPath + '/test')

        # path to covid-19 dataset from https://github.com/ieee8023/covid-chestxray-dataset
        imgPath = self.dataPath + dataSources[0] + '/images'
        csvPath = self.dataPath + dataSources[0] + '/metadata.csv'

        # Path to https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
        kaggle_dataPath = self.dataPath + '/rsna-kaggle-dataset'
        kaggle_csvname = 'stage_2_detailed_class_info.csv'  # get all the normal from here
        kaggle_csvname2 = 'stage_2_train_labels.csv'  # get all the 1s from here since 1 indicate pneumonia
        kaggle_imgPath = 'stage_2_train_images'

        # parameters for COVIDx dataset
        train = []
        test = []
        test_count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
        train_count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}

        mapping = dict()
        mapping['COVID-19'] = 'COVID-19'
        mapping['SARS'] = 'pneumonia'
        mapping['MERS'] = 'pneumonia'
        mapping['Streptococcus'] = 'pneumonia'
        mapping['Normal'] = 'normal'
        mapping['Lung Opacity'] = 'pneumonia'
        mapping['1'] = 'pneumonia'

        # train/test split
        split = 0.1

        # adapted from https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision./datasets.py#L814
        csv = pd.read_csv(csvPath, nrows=None)
        idx_pa = csv["view"] == "PA"  # Keep only the PA view
        csv = csv[idx_pa]

        pneumonias = ["COVID-19", "SARS", "MERS", "ARDS", "Streptococcus"]
        pathologies = ["Pneumonia", "Viral Pneumonia", "Bacterial Pneumonia", "No Finding"] + pneumonias
        pathologies = sorted(pathologies)

        # get non-COVID19 viral, bacteria, and COVID-19 infections from covid-chestxray-dataset
        # stored as patient id, image filename and label
        filename_label = {'normal': [], 'pneumonia': [], 'COVID-19': []}
        count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
        print(csv.keys())
        for index, row in csv.iterrows():
            f = row['finding']
            if f in mapping:
                count[mapping[f]] += 1
                entry = [int(row['patientid']), row['filename'], mapping[f]]
                filename_label[mapping[f]].append(entry)

        print('Data distribution from covid-chestxray-dataset:')
        print(count)

        # add covid-chestxray-dataset into COVIDx dataset
        # since covid-chestxray-dataset doesn't have test dataset
        # split into train/test by patientid
        # for COVIDx:
        # patient 8 is used as non-COVID19 viral test
        # patient 31 is used as bacterial test
        # patients 19, 20, 36, 42, 86 are used as COVID-19 viral test

        for key in filename_label.keys():
            arr = np.array(filename_label[key])
            if arr.size == 0:
                continue
            # split by patients
            # num_diff_patients = len(np.unique(arr[:,0]))
            # num_test = max(1, round(split*num_diff_patients))
            # select num_test number of random patients
            if key == 'pneumonia':
                test_patients = ['8', '31']
            elif key == 'COVID-19':
                test_patients = ['19', '20', '36', '42', '86']  # random.sample(list(arr[:,0]), num_test)
            else:
                test_patients = []
            print('Key: ', key)
            print('Test patients: ', test_patients)
            # go through all the patients
            for patient in arr:
                if patient[0] in test_patients:
                    if COPY_FILE:
                        copyfile(os.path.join(imgPath, patient[1]),
                                 os.path.join(self.dataPath, 'test', patient[1]))
                        test.append(patient)
                        test_count[patient[2]] += 1
                    else:
                        print("WARNING: passing copy file.")
                        break
                else:
                    if COPY_FILE:
                        copyfile(os.path.join(imgPath, patient[1]),
                                 os.path.join(self.dataPath, 'train', patient[1]))
                        train.append(patient)
                        train_count[patient[2]] += 1

                    else:
                        print("WARNING: passing copy file.")
                        break

        print('test count: ', test_count)
        print('train count: ', train_count)

        # add normal and rest of pneumonia cases from https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

        print(kaggle_dataPath)
        csv_normal = pd.read_csv(os.path.join(kaggle_dataPath, kaggle_csvname), nrows=None)
        csv_pneu = pd.read_csv(os.path.join(kaggle_dataPath, kaggle_csvname2), nrows=None)
        patients = {'normal': [], 'pneumonia': []}

        for index, row in csv_normal.iterrows():
            if row['class'] == 'Normal':
                patients['normal'].append(row['patientId'])

        for index, row in csv_pneu.iterrows():
            if int(row['Target']) == 1:
                patients['pneumonia'].append(row['patientId'])

        for key in patients.keys():
            arr = np.array(patients[key])
            if arr.size == 0:
                continue
            # split by patients
            # num_diff_patients = len(np.unique(arr))
            # num_test = max(1, round(split*num_diff_patients))
            # '/content/COVID-Net/'
            test_patients = np.load(self.dataPath + '/COVID-Net/rsna_test_patients_{}.npy'
                                                    ''.format(key))  # random.sample(list(arr), num_test)
            # np.save('rsna_test_patients_{}.npy'.format(key), np.array(test_patients))
            for patient in arr:
                ds = dicom.dcmread(os.path.join(kaggle_dataPath, kaggle_imgPath, patient + '.dcm'))
                pixel_array_numpy = ds.pixel_array
                imgname = patient + '.png'
                if patient in test_patients:
                    if COPY_FILE:
                        cv2.imwrite(os.path.join(self.dataPath, 'test', imgname), pixel_array_numpy)
                        test.append([patient, imgname, key])
                        test_count[key] += 1
                    else:
                        print("WARNING: passing copy file.")
                        break
                else:
                    if COPY_FILE:
                        cv2.imwrite(os.path.join(self.dataPath, 'train', imgname), pixel_array_numpy)
                        train.append([patient, imgname, key])
                        train_count[key] += 1
                    else:
                        print("WARNING: passing copy file.")
                        break
        print('test count: ', test_count)
        print('train count: ', train_count)

        # final stats
        print('Final stats')
        print('Train count: ', train_count)
        print('Test count: ', test_count)
        print('Total length of train: ', len(train))
        print('Total length of test: ', len(test))

        # export to train and test csv
        # format as patientid, filename, label - separated by a space
        train_file = open(self.dataPath + "/train_split_v2.txt", "w")
        for sample in train:
            info = str(sample[0]) + ' ' + sample[1] + ' ' + sample[2] + '\n'
            train_file.write(info)
        train_file.close()

        test_file = open(self.dataPath + "/test_split_v2.txt", "w")
        for sample in test:
            info = str(sample[0]) + ' ' + sample[1] + ' ' + sample[2] + '\n'
            test_file.write(info)
        test_file.close()
