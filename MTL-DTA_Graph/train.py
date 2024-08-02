# %%
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import math
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import argparse

from metrics import get_cindex
from dataset import *
from model import MGraphDTA
from utils import *
from log.train_logger import TrainLogger

def val(model, criterion, dataloader, device, name):
    model.eval()
    running_loss = AverageMeter()

    for data in dataloader:
        data = data.to(device)

        with torch.no_grad():
            if(name=="kd"):
                pred,_,_ = model(data)
            elif(name=="ki"):
                _,pred,_ = model(data)
            else:
                _,_,pred = model(data)
            loss = criterion(pred.view(-1), data.y.view(-1))
            label = data.y
            running_loss.update(loss.item(), label.size(0))

    epoch_loss = running_loss.get_average()
    running_loss.reset()

    model.train()

    return epoch_loss

def main():
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--save_model', action='store_true', help='whether save model or not')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    args = parser.parse_args()

    params = dict(
        data_root="data",
        save_dir ="save",
        save_model=args.save_model,
        lr=args.lr,
        batch_size=args.batch_size,
    )

    logger = TrainLogger(params)
    logger.info(__file__)

    #DATASET = params.get("dataset")
    save_model = params.get("save_model")
    data_root  = params.get("data_root")


    fpath_kd = os.path.join(data_root, "davis")
    fpath_ki = os.path.join(data_root, "drd2")
    fpath_kiba = os.path.join(data_root, "kiba")

    train_kd_set = GNNDataset(fpath_kd, train=True)
    test_kd_set  = GNNDataset(fpath_kd, train=False)

    train_ki_set = GNNDataset(fpath_ki, train=True)
    test_ki_set  = GNNDataset(fpath_ki, train=False)

    train_kiba_set = GNNDataset(fpath_kiba, train=True)
    test_kiba_set  = GNNDataset(fpath_kiba, train=False)

    print(len(train_kd_set), len(train_ki_set), len(train_kiba_set))
    print(len(test_kd_set),  len(test_ki_set),  len(test_kiba_set))

    train_kd_loader = DataLoader(train_kd_set, batch_size=128, shuffle=True, num_workers=8)
    test_kd_loader  = DataLoader(test_kd_set, batch_size=128, shuffle=False, num_workers=8)

    train_ki_loader = DataLoader(train_ki_set, batch_size=128, shuffle=True, num_workers=8)
    test_ki_loader  = DataLoader(test_ki_set, batch_size=128, shuffle=False, num_workers=8)

    train_kiba_loader = DataLoader(train_kiba_set, batch_size=128, shuffle=True, num_workers=8)
    test_kiba_loader  = DataLoader(test_kiba_set, batch_size=128, shuffle=False, num_workers=8)

    device = torch.device('cuda:0')
    model = MGraphDTA(3, 25 + 1, embedding_size=128, filter_num=32, out_dim=1).to(device)
    # model.load_state_dict(torch.load("save/20221103_185954_multi_task/model/epoch-2984, train_loss-0.0193, cindex-0.9722, test_kd_loss-0.2221, test_ki_loss-0.0772, test_kiba_loss-0.0858, average_test_loss-0.1284.pt"))
    epochs = 3000
    steps_per_epoch = 50
    num_iter = math.ceil((epochs * steps_per_epoch) / len(train_kd_loader))
    break_flag = False

    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.MSELoss()

    global_step = 0
    global_epoch = 0
    early_stop_epoch = 400

    running_loss = AverageMeter()
    running_cindex = AverageMeter()
    running_best_mse = BestMeter("min")

    model.train()

    for i in range(num_iter):
        if break_flag:
            break

        for (kd_data, ki_data, kiba_data) in zip(train_kd_loader, train_ki_loader, train_kiba_loader):

            global_step += 1

            #kd_train      
            kd_data = kd_data.to(device)
            pred,_,_ = model(kd_data)

            kd_loss = criterion(pred.view(-1), kd_data.y.view(-1))
            kd_cindex = get_cindex(kd_data.y.detach().cpu().numpy().reshape(-1), pred.detach().cpu().numpy().reshape(-1))

            #ki_train      
            ki_data   = ki_data.to(device)
            _,pred,_  = model(ki_data)

            ki_loss   = criterion(pred.view(-1), ki_data.y.view(-1))
            ki_cindex = get_cindex(ki_data.y.detach().cpu().numpy().reshape(-1), pred.detach().cpu().numpy().reshape(-1))

            #kiba_train      
            kiba_data   = kiba_data.to(device)
            _,_,pred  = model(kiba_data)

            kiba_loss   = criterion(pred.view(-1), kiba_data.y.view(-1))
            kiba_cindex = get_cindex(kiba_data.y.detach().cpu().numpy().reshape(-1), pred.detach().cpu().numpy().reshape(-1))

            loss   = (kd_loss + ki_loss + kiba_loss)/3
            cindex = (kd_cindex + ki_cindex + kiba_cindex)/3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.update(loss.item(), kd_data.y.size(0)) 
            running_cindex.update(cindex, kd_data.y.size(0))

            if global_step % steps_per_epoch == 0:

                global_epoch += 1

                epoch_loss = running_loss.get_average()
                epoch_cindex = running_cindex.get_average()
                running_loss.reset()
                running_cindex.reset()

                test_kd_loss = val(model, criterion, test_kd_loader, device,"kd")
                test_ki_loss = val(model, criterion, test_ki_loader, device,"ki")
                test_kiba_loss = val(model, criterion, test_kiba_loader, device,"kiba")

                average_test_loss = (test_kd_loss + test_ki_loss + test_kiba_loss)/3

                msg = "epoch-%d, train_loss-%.4f, cindex-%.4f, test_kd_loss-%.4f, test_ki_loss-%.4f, test_kiba_loss-%.4f, average_test_loss-%.4f" % (global_epoch, epoch_loss, epoch_cindex, test_kd_loss, test_ki_loss, test_kiba_loss, average_test_loss)
                logger.info(msg)

                if average_test_loss < running_best_mse.get_best():
                    running_best_mse.update(average_test_loss)
                    if save_model:
                        save_model_dict(model, logger.get_model_dir(), msg)
                else:
                    count = running_best_mse.counter()
                    if count > early_stop_epoch:
                        logger.info(f"early stop in epoch {global_epoch}")
                        break_flag = True
                        break
                if global_epoch > epochs:
                    break_flag = True
                    break

if __name__ == "__main__":
    main()
