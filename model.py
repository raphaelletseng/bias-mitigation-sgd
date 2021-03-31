import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
from skorch import NeuralNetClassifier
import pandas as pd
import numpy as np
from skorch.utils import check_indexing
from skorch.utils import multi_indexing
from skorch.utils import to_numpy
from skorch.utils import is_pandas_ndframe
from skorch.utils import flatten
from functools import partial
from scipy import sparse

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
        # Split one output into two
    #    x_cat = D1.get_X_cat()
#        x_cont = D1.get_X_cont()


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

# nope to tuples
# A class for sklearns predict and fit in pytorch
class SampleWeightNeuralNet(NeuralNetClassifier):
    def __init__(self, *args, criterion__reduce=False, **kwargs):
    #def __init__(self, *args, criterion__reduce=False, **kwargs):
        #super().__init__(*args, criterion__reduce = criterion__reduce, **kwargs)
        super().__init__(*args, criterion__reduce=criterion__reduce, **kwargs)



    def fit(self, X_cat, X_cont, y, sample_weight=None):
        X = torch.cat([X_cat, X_cont], 1)
    #    X = D1.get_X()
    #    y = D1.get_y()
    #    X_cat = D1.get_X_cat()
    #    X_cont = D1.get_X_cont()

        if isinstance(X, (pd.DataFrame, pd.Series)) :
            #//category and data point
    #    X_tuple = (X_cat, X_cont)
        #cat_array = X_tuple[0].numpy()
        #cont_array = X_tuple[1].numpy()
        #X_tuple = (cat_array, cont_array)#

            #X_cat = X_cat.to_numpy().astype('float32')
            X = X.to_numpy().astype('float32')
        #if isinstance(X_cont, (pd.DataFrame, pd.Series)):
        #    X_cont = X_cont.to_numpy().astype('float32')
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.to_numpy()
        if sample_weight is not None and isinstance(sample_weight, (pd.DataFrame, pd.Series)):
            sample_weight = sample_weight.to_numpy()
        y = y.reshape([-1,1])
        sample_weight = sample_weight if sample_weight is not None else np.ones_like(y)
        #X_cat = {'X':X_cat, 'sample_weight': sample_weight}
        #X_cont = {'X':X_cont, 'sample_weight': sample_weight}
        #X = {**X_cat, **X_cont}
        X = {'X':X, 'sample_weight': sample_weight}
        return super().fit(X, y)

    def predict(self, X):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.to_numpy().astype('float32')
        return (super().predict_proba(X) > 0.5).astype(np.float)

    def get_loss(self, y_pred, y_true, X, *args, **kwargs):
        loss_unreduced = super().get_loss(y_pred, y_true.float(), X, *args, **kwargs)
        sample_weight = X['sample_weight']
        sample_weight = sample_weight.to(loss_unreduced.device).unsqueeze(-1)
        #sample weights on GPU
        loss_reduced = (sample_weight * loss_unreduced).mean()
        return loss_reduced

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

def _apply_to_data(data, func, unpack_dict=False):
    """Apply a function to data, trying to unpack different data
    types.
    """
    apply_ = partial(_apply_to_data, func=func, unpack_dict=unpack_dict)

    if isinstance(data, dict):
        if unpack_dict:
            return [apply_(v) for v in data.values()]
        return {k: apply_(v) for k, v in data.items()}

    if isinstance(data, (list, tuple)):
        try:
            # e.g.list/tuple of arrays
            return [apply_(x) for x in data]
        except TypeError:
            return func(data)

    return func(data)

def _is_sparse(x):
    try:
        return sparse.issparse(x) or x.is_sparse
    except AttributeError:
        return False

def _len(x):
    if _is_sparse(x):
        return x.shape[0]
    return len(x)

def get_len(data):
    lens = [_apply_to_data(data, _len, unpack_dict=True)]
    lens = list(flatten(lens))
    len_set = set(lens)
    if len(len_set)!=1:
        raise ValueError("Dataset doesn't have consistent lengths")
    return list(len_set)[0]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X_cat, X_cont, y, length= None):
        self.X_cat = X_cat
        self.X_cont = X_cont
        self.y = y

        X = torch.cat([X_cat, X_cont], 1)
        print(X)
        print("#######   THIS IS X ^^^ #########")
        self.X = X
        self.X_indexing = check_indexing(X)

        self.y_indexing = check_indexing(y)

    #    self.X_cat_indexing = check_indexing(X_cat)
    #    self.X_cont_indexing = check_indexing(X_cont)
    #    self.X_cat_is_ndframe = is_pandas_ndframe(X_cat)
    #    self.X_cont_is_ndframe = is_pandas_ndframe(X_cont)
        self.X_is_ndframe = is_pandas_ndframe(X)


        if length is not None:
            self._len = length
            return

        len_X = get_len(X)
        #len_X_cont = get_len(X_cont)
        if y is not None:
            len_y = get_len(y)
            if len_y!= len_X:
                print("len_y: ", len_y)
                print("\n")
                print("len_X: ", len_X)
                raise ValueError("Xs and y have inconsistent lengths")
        self._len = len_X

    def __len__(self):
        return len(self._len)

    def transform(self, X, y):
        y = torch.Tensor([0]) if y is None else y
        if sparse.issparse(X):
            X = X.toarray().squeeze(0)
        return X, y

    def __getitem__(self, i):
        X, y = self.X, self.y
        if self.X_is_ndframe:
            X = {k: X[k].values.reshape(-1,1) for k in X}
        #if self.X_cont_is_ndframe:
        #    X_cont = {k:X_cont[k].values.reshape(-1,1) for k in X_cont}
        Xi = multi_indexing(X, i, self.X_indexing)
        #X_conti = multi_indexing(X_cont, i, self.X_cont_indexing)
        yi = multi_indexing(y, i, self.y_indexing)
        return self.transform(Xi, yi)

    def get_X(self):
        return(self.X)

    def get_y(self):
        return(self.y)

    def get_X_cat(self):
        return (self.X_cat)

    def get_X_cont(self):
        return (self.X_cont)
