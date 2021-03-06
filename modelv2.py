import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
from skorch import NeuralNetClassifier
import pandas as pd
import numpy as np

class RegressionModel(nn.Module):
#class RegressionModel(pl.LightningModule):
    def __init__(self, emb_szs, n_cont, emb_drop, out_sz, szs, drops, y_range, use_bn=True):
        super().__init__()

        # embeddings
        for i, (c, s) in enumerate(emb_szs): assert c > 1, f"cardinality must be >=2, got emb_szs[{i}]: ({c},{s})"
        self.embs = nn.ModuleList([nn.Embedding(c+1, s) for c, s in emb_szs])

        for emb in self.embs: emb_init(emb)
        n_emb = sum(e.embedding_dim for e in self.embs)
        self.n_emb, self.n_cont = n_emb, n_cont


        # linear & batch norm/group norm)
        szs = [n_emb + n_cont] + szs
        self.lins = nn.ModuleList([nn.Linear(szs[i], szs[i + 1]) for i in range(len(szs) - 1)])
        self.bns = nn.ModuleList([nn.GroupNorm(1, sz) for sz in szs[1:]])
        for o in self.lins: nn.init.kaiming_normal_(o.weight.data)
        self.outp = nn.Linear(szs[-1], out_sz)
        nn.init.kaiming_normal_(self.outp.weight.data)

        # output
        self.emb_drop = nn.Dropout(emb_drop)
        self.drops = nn.ModuleList([nn.Dropout(drop) for drop in drops])
        self.bn = nn.GroupNorm(1,n_cont)
        self.use_bn, self.y_range = use_bn, y_range
        self.activation = nn.Sigmoid()

    def forward(self, x_cat, x_cont):


        # embedding for categorical variables
        if self.n_emb != 0:
            x = [e(x_cat[:, i]) for i, e in enumerate(self.embs)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)

        # embedding for continuous variables
        if self.n_cont != 0:
            x2 = self.bn(x_cont.float())
            x = torch.cat([x, x2], 1) if self.n_emb != 0 else x2
        for l, d, b in zip(self.lins, self.drops, self.bns):
            x = F.relu(l(x))
            if self.use_bn: x = b(x)
            x = d(x)

        # regression layer
        x = self.outp(x)
        if self.y_range:
            x = self.activation(x)
            x = x * (self.y_range[1] - self.y_range[0])
            x = x + self.y_range[0]
#            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
#            x = torch.where(torch.isinf(x), torch.zeros_like(x), x)
        return x.squeeze()


    def configure_optimizers(self):
        #        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
        return optimizer

    def BCELoss(self, output, target):
        return nn.BCELoss()

    def name(self):
        return "RegressionModel"


    def emb_init(x):
        x = x.weight.data
        sc = 2/(x.size(1)+1)
        x.uniform_(-sc,sc)

# A class for sklearns predict and fit in pytorch
net = NeuralNetClassifier(
    RegressionModel,
    max_epochs = 10,
    lr = 0.1,
    iterator_train_shuffle=True,
)

net.fit(x_cat + x_cont, y)
yprob = net.predict_proba(x_cat+x_cont)

'''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = SampleWeightNeuralNet(
        RegressionModel,
        max_epochs = 20,
    #optimizer = optim.Adam,
        lr = 0.001,
    #batch_size = 512,
    #train_split = None,
        iterator_train_shuffle = True,
        criterion = nn.BCELoss,
        device = device )

    def fit(X, y):
        fit = net.fit(X,y)
        return fit

    def pred(y):
        y_pred = net.predict(X)
        return y_pred
'''

def emb_init(x):
    x = x.weight.data
    sc = 2/(x.size(1)+1)
    x.uniform_(-sc,sc)
