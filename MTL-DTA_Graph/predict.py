# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import argparse

from metrics import get_cindex, get_rm2
from dataset import *
from model import MGraphDTA
from utils import *

def val(model, dataloader, device):
    model.eval()
    
    kd_list = []
    ki_list = []
    kiba_list = []

    for data in dataloader:
        data = data.to(device)

        with torch.no_grad():
            kd,ki,kiba = model(data)
            kd_list.append(kd.detach().cpu().numpy().reshape(-1))
            ki_list.append(ki.detach().cpu().numpy().reshape(-1))
            kiba_list.append(kiba.detach().cpu().numpy().reshape(-1))

    kd_list = [item for sublist in kd_list for item in sublist]
    ki_list = [item for sublist in ki_list for item in sublist]
    kiba_list = [item for sublist in kiba_list for item in sublist]

    return kd_list, ki_list, kiba_list

def main():
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--dataset', required=True, help='davis or kiba')
    parser.add_argument('--model_path', required=True, type=str, help='model path ready to load')
    args = parser.parse_args()

    data_root = "data"
    DATASET = args.dataset
    model_path = args.model_path
    fpath = os.path.join(data_root, DATASET)

    test_set = GNNDataset(fpath, name=DATASET)
    print("Number of test: ", len(test_set))
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=8)

    device = torch.device('cuda')
    model = MGraphDTA(3, 25 + 1, embedding_size=128, filter_num=32, out_dim=1).to(device)

    load_model_dict(model, model_path)
    kd_list, ki_list, kiba_list = val(model, test_loader, device)


    df = pd.read_csv("data/raw/{}.csv".format(DATASET))
    df["kd"] = kd_list
    df["ki"] = ki_list
    df["kiba"] = kiba_list

    df.to_csv("data/predictions/{}_pred.csv".format(DATASET))

    msg = "{} dataset predictions are saved".format(DATASET)
    print(msg)


if __name__ == "__main__":
    main()