import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from keras_preprocessing.sequence import pad_sequences

def smiles_encode(smiles, smiles_dict):
    output = []
    for c in smiles:
        # print(c)
        output.append(smiles_dict.get(c))
    return output

def protein_encode(protein, protein_dict):
    output = []
    for c in protein:
        # print(c)
        output.append(protein_dict.get(c))
    return output

def main():
    parser = argparse.ArgumentParser()

    #arguments
    parser.add_argument('--data_path', default="/", help='path for datasets')

    args = parser.parse_args()

    data_path = args.data_path

    print("\ndataset path: {}".format(data_path))

    #datasets
    kiba  = pd.read_csv(os.path.join(data_path,"kiba/raw/data.csv"))
    metz  = pd.read_csv(os.path.join(data_path,"metz/raw/data.csv"))
    davis = pd.read_csv(os.path.join(data_path,"davis/raw/data.csv"))

    df = pd.concat([kiba, metz, davis])

    smiles = df["compound_iso_smiles"].to_list()
    smiles_chars = ['[PAD]'] + list(set([ c for compounds in smiles for c in compounds ]))
    smiles_dict = { c:i for i,c in enumerate(smiles_chars) }

    targets = df["target_sequence"].to_list()
    protein_chars = ['[PAD]'] + list(set([ c for compounds in targets for c in compounds ]))
    protein_dict = { c:i for i,c in enumerate(protein_chars) }

    print("preprocessing started...")
    dataset_dict = {0:"kiba", 1:"metz", 2:"davis"}


    for i, dataset in enumerate([kiba, metz, davis]):
        # creating dir to save fold_setting
        data_path = os.path.join(data_path,"{}/raw".format(dataset_dict[i]))
        if(os.path.exists(data_path) != True):
            os.makedirs(data_path)

        # save train and test split
        data, test = train_test_split(dataset, test_size=0.2)

        data = data.reset_index(drop=True)
        test  = test.reset_index(drop=True)

        data.to_csv(os.path.join(data_path,"{}/raw/data_train.csv".format(dataset_dict[i])))
        test.to_csv(os.path.join(data_path,"{}/raw/data_test.csv".format(dataset_dict[i])))
            

    for i in len(dataset_dict):

        pro_path = os.path.join(data_path,"{}/processed".format(dataset_dict[i]))
            
        train = pd.read_csv(os.path.join(data_path,"{}/raw/data_train.csv".format(dataset_dict[i])),sep=',', header=None)
        test = pd.read_csv(os.path.join(data_path,"{}/raw/data_test.csv".format(dataset_dict[i])),sep=',', header=None)
        
            
        print("lengh of train:{}".format(len(train)))
        print("lengh of test:{}".format(len(test)))

        #preprocessing

        train_tokens_d = torch.tensor(pad_sequences([ smiles_encode(compound, smiles_dict) for compound in train['compound_iso_smiles']], maxlen=100, padding='post', truncating='post'))
        test_tokens_d  = torch.tensor(pad_sequences([ smiles_encode(compound, smiles_dict) for compound in test['compound_iso_smiles']], maxlen=100, padding='post', truncating='post'))

        train_tokens_p = torch.tensor(pad_sequences([ protein_encode(protein, protein_dict) for protein in train['target_sequence']], maxlen=1200, padding='post', truncating='post'))
        test_tokens_p  = torch.tensor(pad_sequences([ protein_encode(protein, protein_dict) for protein in test['target_sequence']], maxlen=1200, padding='post', truncating='post'))

        y_train        = torch.tensor(dataset.iloc[train]["affinity"].to_list())
        y_test         = torch.tensor(dataset.iloc[test]["affinity"].to_list())

        train_set = TensorDataset(train_tokens_d, train_tokens_p, y_train)
        test_set  = TensorDataset(test_tokens_d, test_tokens_p, y_test)

        torch.save(train_set, os.path.join(pro_path,"train_set.pth"))
        torch.save(test_set, os.path.join(pro_path,"test_set.pth"))

    print("preprocessing finished\n")

if __name__ == "__main__":
    main()

