import torch
from torch import nn
import pytorch_lightning as pl

class DeepDTA(pl.LightningModule):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128,num_features_xd=36, num_features_xp=22, output_dim=128, dropout=0.1):
        super(DeepDTA, self).__init__()
        
        #smile layers
        self.drug_embedding = nn.Embedding(num_features_xd, embed_dim)
        self.dconv1         = nn.Conv1d(embed_dim, n_filters, 4)
        self.dconv2         = nn.Conv1d(n_filters, n_filters*2, 6)
        self.dconv3         = nn.Conv1d(n_filters*2, n_filters*3, 8)
        self.dmaxpool       = nn.AdaptiveMaxPool1d(1)
        
        #protein layers
        self.prot_embedding = nn.Embedding(num_features_xp, embed_dim)
        self.pconv1         = nn.Conv1d(embed_dim, n_filters, 4)
        self.pconv2         = nn.Conv1d(n_filters, n_filters*2, 8)
        self.pconv3         = nn.Conv1d(n_filters*2, n_filters*3, 12)
        self.pmaxpool       = nn.AdaptiveMaxPool1d(1)
        
        #combined layers
        self.comb_lin1      = nn.Linear(n_filters*6, 1024)
        self.comb_lin2      = nn.Linear(1024, 1024)
        self.comb_lin3      = nn.Linear(1024, 512)
        self.out_layer1     = nn.Linear(512, n_output)
        self.out_layer2     = nn.Linear(512, n_output)
         
        #other layers
        self.dropout        = nn.Dropout(dropout)
        self.relu           = nn.ReLU()
    

    # def training_step(self, XD, XP, Y):
    def forward(self, X):
        # print(len(X))
        (XD, XP, Y) = X
        # print(XD.size())
        
        #smile
        drug_em  = self.drug_embedding(XD).permute(0,2,1)
        drug_em  = self.relu(self.dconv1(drug_em))
        drug_em  = self.relu(self.dconv2(drug_em))
        drug_em  = self.relu(self.dconv3(drug_em))
        drug_out = self.dmaxpool(drug_em)
        
        #protein
        prot_em  = self.prot_embedding(XP).permute(0,2,1)
        prot_em  = self.relu(self.pconv1(prot_em))
        prot_em  = self.relu(self.pconv2(prot_em))
        prot_em  = self.relu(self.pconv3(prot_em))
        prot_out = self.pmaxpool(prot_em)
        
        #compbined
        dp_em   = torch.cat([drug_out, prot_out],2)
        dp_em   = dp_em.view(dp_em.shape[0], -1)
        
        dp_em   = self.dropout(self.relu(self.comb_lin1(dp_em)))
        dp_em   = self.dropout(self.relu(self.comb_lin2(dp_em)))
        dp_em   = self.dropout(self.relu(self.comb_lin3(dp_em)))
        
        out1     = self.out_layer1(dp_em)
        out2     = self.out_layer2(dp_em)


        return out1, out2