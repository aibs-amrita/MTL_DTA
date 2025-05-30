# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
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
from model2 import MGraphDTA
from utils import *
from log.train_logger import TrainLogger

def val(model, criterion, dataloader, device, name):
    model.eval()
    running_loss = AverageMeter()

    for data in dataloader:
        data = data.to(device)

        with torch.no_grad():
            if(name=="kiba"):
                pred,_ = model(data)
            elif(name=="davis"):
                _,pred= model(data)

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
        save_dir="save/k-d",
        save_model=args.save_model,
        lr=args.lr,
        batch_size=args.batch_size
    )

    logger = TrainLogger(params)
    logger.info(__file__)

    #DATASET = params.get("dataset")
    save_model = params.get("save_model")
    data_root = params.get("data_root")

    fpath_kiba = os.path.join(data_root, "kiba")
    fpath_davis = os.path.join(data_root, "davis_kd")
    # fpath_ic50 = os.path.join(data_root, "davisba")

    train_kiba_set = GNNDataset(fpath_kiba, train=True)
    test_kiba_set  = GNNDataset(fpath_kiba, train=False)

    train_davis_set = GNNDataset(fpath_davis, train=True)
    test_davis_set  = GNNDataset(fpath_davis, train=False)

    # train_ic50_set = GNNDataset(fpath_ic50, train=True)
    # test_ic50_set  = GNNDataset(fpath_ic50, train=False)

    print(len(train_kiba_set), len(train_davis_set))
    print(len(test_kiba_set),  len(test_davis_set))

    train_kiba_loader = DataLoader(train_kiba_set, batch_size=128, shuffle=True, num_workers=8)
    test_kiba_loader  = DataLoader(test_kiba_set, batch_size=128, shuffle=False, num_workers=8)

    train_davis_loader = DataLoader(train_davis_set, batch_size=128, shuffle=True, num_workers=8)
    test_davis_loader  = DataLoader(test_davis_set, batch_size=128, shuffle=False, num_workers=8)

    device = torch.device('cuda:0')
    model = MGraphDTA(3, 25 + 1, embedding_size=128, filter_num=32, out_dim=1).to(device)
    # model.load_state_dict(torch.load('/home/gopichand/Multi-task_model/save/20220627_185930_multi_task/model/epoch-2739, train_loss-0.0216, cindex-0.9711, test_kiba_loss-0.2211, test_davis_loss-0.0864, test_davisba_loss-0.0878, average_test_loss-0.1318.pt'))
    epochs = 3000
    steps_per_epoch = 50
    num_iter = math.ceil((epochs * steps_per_epoch) / len(train_davis_loader))
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
            logger.info(f"break_flag{global_epoch}")
            break

        for (kiba_data, davis_data) in zip(train_kiba_loader, train_davis_loader):

            global_step += 1

            #kiba_train      
            kiba_data = kiba_data.to(device)
            pred,_ = model(kiba_data)

            kiba_loss = criterion(pred.view(-1), kiba_data.y.view(-1))
            kiba_cindex = get_cindex(kiba_data.y.detach().cpu().numpy().reshape(-1), pred.detach().cpu().numpy().reshape(-1))

            #davis_train      
            davis_data   = davis_data.to(device)
            _,pred  = model(davis_data)

            davis_loss   = criterion(pred.view(-1), davis_data.y.view(-1))
            davis_cindex = get_cindex(davis_data.y.detach().cpu().numpy().reshape(-1), pred.detach().cpu().numpy().reshape(-1))

            loss   = (kiba_loss + davis_loss)/2
            cindex = (kiba_cindex + davis_cindex)/2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.update(loss.item(), kiba_data.y.size(0)) 
            running_cindex.update(cindex, kiba_data.y.size(0))

            if global_step % steps_per_epoch == 0:

                global_epoch += 1

                epoch_loss = running_loss.get_average()
                epoch_cindex = running_cindex.get_average()
                running_loss.reset()
                running_cindex.reset()

                test_kiba_loss = val(model, criterion, test_kiba_loader, device,"kiba")
                test_davis_loss = val(model, criterion, test_davis_loader, device,"davis")
                # test_ic50_loss = val(model, criterion, test_ic50_loader, device,"davisba")

                average_test_loss = (test_kiba_loss + test_davis_loss)/2

                msg = "epoch-%d, train_loss-%.4f, cindex-%.4f, test_kiba_loss-%.4f, test_davis_loss-%.4f, average_test_loss-%.4f" % (global_epoch, epoch_loss, epoch_cindex, test_kiba_loss, test_davis_loss, average_test_loss)
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
                    logger.info(f"training finished{global_epoch}")
                    break_flag = True
                    break

if __name__ == "__main__":
    main()
