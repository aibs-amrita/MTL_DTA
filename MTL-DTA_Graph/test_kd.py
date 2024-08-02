# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import argparse

from metrics import get_cindex, get_rm2
from dataset import *
from model2 import MGraphDTA
from utils import *

def val(model, criterion, dataloader, device, name):
    model.eval()
    running_loss = AverageMeter()
    running_ci   = AverageMeter()
    running_r2   = AverageMeter()

    pred_list = []
    label_list = []

    for data in dataloader:
        data = data.to(device)

        with torch.no_grad():
            if(name=="kiba"):
                pred,_ = model(data)
            else:
                _,pred= model(data)
            loss = criterion(pred.view(-1), data.y.view(-1))
            label = data.y
            running_loss.update(loss.item(), label.size(0))
            running_ci.update(get_cindex(data.y.detach().cpu().numpy().reshape(-1), pred.detach().cpu().numpy().reshape(-1)))
            running_r2.update(get_rm2(data.y.detach().cpu().numpy().reshape(-1), pred.detach().cpu().numpy().reshape(-1)))
            pred_list.extend(pred.detach().cpu().numpy().reshape(-1))
            label_list.extend(label.detach().cpu().numpy().reshape(-1))
  
    epoch_r2     = running_r2.get_average()
    epoch_cindex = running_ci.get_average()
    epoch_loss   = running_loss.get_average()
    running_loss.reset()
    running_ci.reset()
    running_r2.reset()

    return epoch_loss, epoch_cindex, epoch_r2, pred_list, label_list

def main():
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--dataset', required=True, help='kiba or metz')
    parser.add_argument('--model_path', required=True, type=str, help='model path ready to load')
    args = parser.parse_args()

    data_root = "data/test_data"
    DATASET = args.dataset
    model_path = args.model_path

    fpath = os.path.join(data_root, DATASET)

    test_set = GNNDataset(fpath, train=False)
    print("Number of test: ", len(test_set))
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=8)

    device = torch.device('cuda:0')
    model = MGraphDTA(3, 25 + 1, embedding_size=128, filter_num=32, out_dim=1).to(device)

    criterion = nn.MSELoss()
    load_model_dict(model, model_path)
    test_loss, test_cindex, test_r2, pred_list, label_list = val(model, criterion, test_loader, device, args.dataset)
    msg = "test_loss:%.4f, test_cindex:%.4f, test_r2:%.4f" % (test_loss, test_cindex, test_r2)
    np.savetxt("pred.txt", pred_list)
    np.savetxt("label.txt", label_list)
    print(msg)


if __name__ == "__main__":
    main()

# %%
