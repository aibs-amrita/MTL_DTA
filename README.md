# Enhancing Drug Target Affinity Prediction Utilizing Graph and Sequence-Based Deep Learning Models with A Multitask Learning Framework

## Abstarct

Drug target affinity is an important measure to identify potential interacting partners for a target of interest in the early stages of drug discovery. However, the high research cost and time-consuming nature of the traditional drug discovery approaches make them a less attractive choice when used solely. Deep learning approaches are emerging as an alternative strategy to be used for accelerating the early stages of drug discovery. Existing deep learning approaches for drug target affinity prediction methods were able to contribute to solving the problem of drug target affinity prediction using individual datasets. However, the datasets available for training these models were limited in their diversity, which hinders the performance of models developed using these datasets. This study proposes a novel multitask learning approach to develop drug target affinity prediction models. Multitask leaning approach involves training the model with multiple datasets instead of a single dataset. In this study, multitask learning approach was utilized to develop graph-based and sequence-based drug target affinity prediction models. Results demonstrate that this approach increased the performance of KiBA and Metz datasets in both sequence-based and graph-based models.  Furthermore, to understand the effects of individual datasets on the performance of other datasets, pairwise combinations of datasets were utilized for developing models in a multitask approach. Results from the pairwise combination method indicate that the selection of the right combination of the datasets to develop multitask learning models is critical and will positively influence the performance of the drug target affinity prediction models.

![Alt text](graphical_abstract.png "MTL_DTA")

## Requirements

matplotlib >= 3.2.2

pandas >= 1.2.4

torch_geometric >= 1.7.0

torch >= 1.7.1

pytorch_lightning >= 2.3.0

tqdm >= 4.51.0

networkx >= 2.5.1

numpy >= 1.20.1

ipython >= 7.24.1

rdkit >= 2009.Q1-1

scikit_learn >= 0.24.2

## Description of folders

* **MTL-DTA_Graph** : Folder includes source code for the Graph based MTL3 and MTL2 models

    + **data** folder contains raw data of Kiba, metz and davis datasets.
    + **log** folder includes the source codes to record the training process.
    + **dataset.py** file prepares the data for training.
    + **metrics.py** contains a series of metrics to evalute the model performances.
    + **model.py**, the implementation of MTL3-DTA_Graph can be found here.
    + **model2.py**, the implementation of MTL2-DTA_Graph can be found here.
    + **preprocessing.py**, a file that preprocesses the raw data into graph format and should be executed before model trianing.
    + **test.py**, test a trained model and print the results.
    + **train.py**, train MTL-DTA_Graph model.
    + **utils.py** file includes useful tools for model training.

* **MTL-DTA_Seq** : Folder includes source code for the Sequence based MTL3 and MTL2 models

    + **data** folder contains raw data of Kiba, metz and davis datasets.
    + **model.py**, the implementation of MTL3-DTA_Seq can be found here.
    + **model2.py**, the implementation of MTL2-DTA_Seq can be found here.
    + **preprocessing.py**, a file that preprocesses the raw data into  format and should be executed before model trianing.
    + **test.py**, test a trained model and print the results.
    + **train.py**, train MTL-DTA_Seq model.