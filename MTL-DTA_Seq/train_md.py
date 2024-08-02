import os
import time
import torch
import argparse
import torch.nn.functional as F
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset
from lifelines.utils import concordance_index
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.utilities.seed import seed_everything
from model2 import DeepDTA

def main():

    seed_everything(seed = 100, workers=True)

    parser = argparse.ArgumentParser()

    #arguments
    parser.add_argument('--data_path', default="data/", help='path for datasets')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--mode', default="train", help="train or test")
    parser.add_argument('--cpkt', help="cpkt file name for validating")


    args = parser.parse_args()

    data_path  = args.data_path
    batch_size = args.batch_size
    epochs     = args.epochs
    mode       = args.mode
    cpkt       = args.cpkt

    data_module      = data_class(data_path, batch_size)

    train_loader     = data_module.train_dataloader()
    test_loader      = data_module.test_dataloader()

    model            = pl_class()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dirpath   = os.path.join(data_path,"models/m-d", timestamp)

    model_checkpoint = ModelCheckpoint(dirpath, filename="{epoch}_{" + "val_loss}",monitor= 'val_loss', mode='min', save_top_k=3)

    logger           = TensorBoardLogger(os.path.join(data_path,'logs'))

    trainer          = pl.Trainer(gpus= [1],max_epochs=epochs, callbacks=[model_checkpoint], logger=logger)

    if(mode == "train"):
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    else:
        trainer.validate(model, val_dataloaders=test_loader, ckpt_path=os.path.join(data_path, "models", cpkt))

class pl_class(DeepDTA):

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def loss_function(self, y_pred, y_true):
        loss = F.mse_loss(y_pred, y_true)
        return loss
    
    def concordance_index(self, y_true, y_pred):
        ci = concordance_index(y_true, y_pred)
        return ci
    
    def r2_score(self, y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        return r2

    def training_step(self, batch, batch_idx):

        metz  = batch["metz"]
        davis  = batch["davis"]

        #metz
        y_pred,_      = self.forward(metz)
        (_,_,y_true)  = metz
        metz_loss     = self.loss_function(y_pred.view(-1), y_true)
        metz_ci       = self.concordance_index(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())

        # davis
        _,y_pred      = self.forward(davis)
        (_,_,y_true)  = davis
        davis_loss     = self.loss_function(y_pred.view(-1), y_true)
        davis_ci       = self.concordance_index(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())

        avg_loss = (metz_loss + davis_loss)/2
        avg_ci   = (metz_ci + davis_ci)/2

        self.log("train_loss", avg_loss)
        self.log("train_ci", avg_ci)
        return avg_loss

    def validation_step(self, batch, batch_idx):

        metz  = batch["metz"]
        davis  = batch["davis"]

        #metz
        y_pred,_      = self.forward(metz)
        (_,_,y_true)  = metz
        metz_loss     = self.loss_function(y_pred.view(-1), y_true)
        metz_ci       = self.concordance_index(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
        metz_r2       = self.r2_score(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())

        # davis
        _,y_pred      = self.forward(davis)
        (_,_,y_true)  = davis
        davis_loss    = self.loss_function(y_pred.view(-1), y_true)
        davis_ci      = self.concordance_index(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
        davis_r2      = self.r2_score(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())


        avg_loss = (metz_loss + davis_loss)/2
        avg_ci   = (metz_ci + davis_ci)/2

        self.log("val_loss", avg_loss.item(), prog_bar = True)
        self.log("val_ci", avg_ci, prog_bar = True)
        self.log("metz_loss", metz_loss.item())
        self.log("davis_loss", davis_loss.item())
        self.log("metz_ci", metz_ci)
        self.log("davis_ci", davis_ci)
        self.log("metz_r2", metz_r2)
        self.log("davis_r2", davis_r2)


class data_class(pl.LightningModule):

    def __init__(self, data_path, batch_size):
        self.data_path   = data_path
        self.batch_size  = batch_size

        self.train_metz  = torch.load(os.path.join(self.data_path, "metz/processed/train_set.pth"))
        self.train_davis = torch.load(os.path.join(self.data_path, "davis/processed/train_set.pth"))

        self.test_metz   = torch.load(os.path.join(self.data_path, "metz/processed/test_set.pth"))
        self.test_davis  = torch.load(os.path.join(self.data_path, "davis/processed/test_set.pth"))

        

    def train_dataloader(self):
        metz_train_loader  = DataLoader(self.train_metz, batch_size = self.batch_size)
        davis_train_loader = DataLoader(self.train_davis, batch_size = self.batch_size)

        loaders = {"metz": metz_train_loader, "davis": davis_train_loader}

        combined_loaders   = CombinedLoader(loaders, mode="max_size_cycle")

        return combined_loaders
    
    def test_dataloader(self):
        metz_test_loader  = DataLoader(self.test_metz, batch_size = self.batch_size)
        davis_test_loader = DataLoader(self.test_davis, batch_size = self.batch_size)

        loaders = {"metz": metz_test_loader, "davis": davis_test_loader}

        combined_loaders  = CombinedLoader(loaders, mode="max_size_cycle")

        return combined_loaders


if __name__ == "__main__":
    main()