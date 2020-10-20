# Federated Learning:Model Poisoning Attacks in Healthcare

This project constitutes a simulation environment for FL tasks that facilitates evaluating the perfomance of the different robbust and privacy preserving FL techniques against different attack scenrios.
At the current stage, the project tackles several aspects of FL in adversarial settings.

- **Robust Aggregation FL**: currently there are 4 main aggregation schemes implemented: Federated Averaging(not robust), MKRUM, COMED and Adaptive Federated Averaging.

- **Privacy preserving FL**: we have experimented with client level differential privacy and the syntactic approch for FL, that makes use of Information loss to perform generalisation over the training and testing datasets. 

- **FL Adversial stragegies**: We have simulated two different startegies attackers can have: labe-flippinng attacks or byzantine ones, that send noisy parameter updates.
---
## Initial set-up

We recomend using Python 3.6 for compatibility with all the used modules and libraries.
setupVenvOnGPUCluster.sh is the script that can be used for the initial setup of the python virtual enviroment; we have used it for the cloud-based virtual machines and the GPU cluster provided by from Imperial College Cloud

### Using the running scripts

runExperiment.sh and stopLastExperiment.sh can be used for running the script locally or on a virtual machine send write the output to file.
runExperimentGPU.sh can be used for running the experiment using the Slurm GPU Cluster from Imperial College Cloud

## Extending the project

To experiment with new datasets and models there are a few steps that need to be followed. 
Firstly, a new DatasetLoader child class needs to be created for the newly considered dataset. The single public method of this class, representing its only communication bridge to the other modules classes and modules is getDatasets method. This method, given a list of percentages representing the data split among clients, it will return a set of datasets corresponding to the clients' data share, and a test dataset, which is later passed to the aggregator. 

In order to use a new model, this could be added to the classifiers directory within a dedicated module file. Finally, all of those elements could be put together within the main.py experiment setup methods, similarly to the already existing ones.

We also provided a straight forward mechanism of extending the code base to use new aggregation schemes: creating child classes of the Aggregator class in aggregators.py module.
