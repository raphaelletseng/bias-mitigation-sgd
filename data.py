import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sampler import BalancedBatchSampler
import pandas as pd
import numpy as np
import random

pd.set_option('mode.chained_assignment', None)

#collect data from adult-data

class data_loader():
    def __init__(self, args, s):
        print("Args: " + args.dataset)

        train_path = 'adult-data/adult.data'
        test_path = 'adult-data/adult.test'

        cols = ['age', 'workclass', 'fnlwgt', 'education',
                    'education-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain',
                    'capital-loss', 'hours-per-week', 'native-country', 'y']
        train_df = pd.read_csv(train_path, sep=',', names=cols, nrows=150)
        test_df = pd.read_csv(test_path, sep=',', names=cols, nrows=150)

        train_df = train_df.replace({'?': np.nan})
        test_df = test_df.replace({'?': np.nan})

        train_df['y'] = train_df['y'].apply(lambda x: 0 if ">50K" in x else 1)
        test_df['y'] = test_df['y'].apply(lambda x: 0 if ">50K" in x else 1)

        train_df = train_df.dropna()
        test_df = test_df.dropna()

        train_df = train_df.sample(frac=1).reset_index(drop=True)  # shuffle df
        test_df = test_df.sample(frac=1).reset_index(drop=True)

#--------------------------------------------------------------#
        train_data = LoadDataset(train_df, args.dataset, args.sensitive)
        test_data = LoadDataset(test_df, args.dataset, args.sensitive)

        self.sensitive_keys = train_data.getkeys()
        self.train_size = len(train_data)
        self.test_size = len(test_data)
        self.sensitive_col_idx = train_data.get_sensitive_idx()
        self.cat_emb_size = train_data.categorical_embedding_sizes  # size of categorical embedding
        print(self.cat_emb_size)
        self.num_conts = train_data.num_numerical_cols  # number of numerical variables

        class_count = dict(train_df.y.value_counts())
        class_weights = [value / len(train_data) for _, value in class_count.items()]

        train_batch = args.batch_size
        test_batch = len(test_data)
        self.train_loader = DataLoader(dataset=train_data,
                                           sampler=BalancedBatchSampler(train_data, train_data.Y),
                                           batch_size=train_batch)

        self.test_loader = DataLoader(dataset=test_data,
                                          batch_size=test_batch,
                                          drop_last=True)

    def getkeys(self):
        return self.sensitive_keys

    def get_input_properties(self):
        return self.cat_emb_size, self.num_conts

    def __getitem__(self):
        return self.train_loader, self.test_size

    def get_sensitive_idx(self):
        return self.sensitive_col_idx

    def __len__(self):
        return self.train_size, self.test_size



class LoadDataset(Dataset):
    def __init__(self, data, mode, sensitive_col):

        self.len = data.shape[0]

        print(data.head())
        # define data column types
        categorical_columns = ['workclass', 'education', 'marital-status',
                                   'occupation', 'relationship', 'race',
                                   'sex', 'native-country']

        numerical_columns = ['education-num', 'capital-gain',
                                 'capital-loss', 'hours-per-week']

            # categorical variables
        for category in categorical_columns:
            data[category] = data[category].astype('category')

        workclass = data['workclass'].cat.codes.values
        education = data['education'].cat.codes.values
        marital_status = data['marital-status'].cat.codes.values
        occupation = data['occupation'].cat.codes.values
        relationship = data['relationship'].cat.codes.values
        race = data['race'].cat.codes.values
        sex = data['sex'].cat.codes.values
        native_country = data['native-country'].cat.codes.values
        self.cat_dict = dict(enumerate(data[sensitive_col].cat.categories)) # 1
            #self.cat_dict = dict(enumerate(data['race'].cat.categories))  # 5
            #            self.cat_dict = dict(enumerate(data['marital-status'].cat.categories)) # 2
        self.sensitive_col_idx = categorical_columns.index(sensitive_col)

        print(self.cat_dict)

        categorical_data = np.stack([workclass, education, marital_status,
                                         occupation, relationship, race,
                                         sex, native_country], 1)

        self.categorical_data = torch.tensor(categorical_data, dtype = torch.int64)

        for numerical in numerical_columns:
                data[numerical] = pd.to_numeric(data[numerical], errors='coerce')
        numerical_data = np.stack([data[col].values.astype(np.float) for col in numerical_columns], 1)
        self.numerical_data = torch.tensor(numerical_data, dtype=torch.float64)

        # target variable
        data['y'] = data['y'].astype('category')
        Y = data['y'].cat.codes.values
        self.Y = torch.tensor(Y.flatten(), dtype=torch.int64)

        # define categorical and continuous embedding sizes
        categorical_column_sizes = [len(data[column].cat.categories) for column in categorical_columns] #This one might not be necessary
        self.categorical_embedding_sizes = [(col_size, min(50, (col_size + 1) // 2))
                                            for col_size in categorical_column_sizes]

        self.num_numerical_cols = self.numerical_data.shape[1]

    def get_sensitive_idx(self):
        return  self.sensitive_col_idx

    def getkeys(self):
        return self.cat_dict

    def __getitem__(self, idx):
        return self.categorical_data[idx], self.numerical_data[idx].float(), self.Y[idx].float()

    def __len__(self):
        return self.len


def get_class_distribution(dataset_obj):
    count_dict = {k: 0 for k, v in dataset_obj.class_to_idx.items()}

    for element in dataset_obj:
        y_lbl = element[1]
        y_lbl = idx2class[y_lbl]
        count_dict[y_lbl] += 1

    return count_dict
