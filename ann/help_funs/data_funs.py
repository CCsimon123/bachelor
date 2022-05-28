import pandas as pd
from numpy.random import RandomState
import torch

def create_data(path_to_data, train_data_frac, train_path, val_path):
    df = pd.read_csv(path_to_data, names=['SWH', 'mean', 'var', 'azm', 'flag'])
    # Removes null values
    df.drop(df[df['SWH'].isnull()].index, inplace=True)
    rng = RandomState(4)

    # Make a mean of 0 and std of 1 for all input variables
    df['mean'] = (df['mean'] - df['mean'].mean()) / (df['mean'].std())
    df['var'] = (df['var'] - df['var'].mean()) / (df['var'].std())
    df['azm'] = (df['azm'] - df['azm'].mean()) / (df['azm'].std())
    df = df[['mean', 'var', 'azm', 'SWH']]

    # train validation split the data
    train = df.sample(frac=train_data_frac, random_state=rng)
    val = df.loc[~df.index.isin(train.index)]
    # save data in csv-files
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)


# Dataset class to use csv data in a neural network
class CustomCsvDataset:
    def __init__(self, dataset, device):
        self.dataset = dataset.to(device)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_data = self.dataset[idx, 0:(self.dataset.size(1) - 1)]
        label = self.dataset[idx, self.dataset.size(1) - 1]
        return input_data, label


# the neural network
class Net(torch.nn.Module):
    def __init__(self, num_of_input):
        super(Net, self).__init__()
        self.hid1 = torch.nn.Linear(num_of_input, 30)
        self.hid2 = torch.nn.Linear(30, 30)
        self.hid3 = torch.nn.Linear(30, 30)
        self.hid4 = torch.nn.Linear(30, 30)
        self.hid5 = torch.nn.Linear(30, 30)
        self.hid6 = torch.nn.Linear(30, 30)
        self.hid7 = torch.nn.Linear(30, 30)
        self.hid8 = torch.nn.Linear(30, 30)
        self.output = torch.nn.Linear(30, 1)

    def forward(self, x):
        z = torch.relu(self.hid1(x))
        z = torch.relu(self.hid2(z))
        z = torch.relu(self.hid3(z))
        z = torch.relu(self.hid4(z))
        z = torch.relu(self.hid5(z))
        z = torch.relu(self.hid6(z))
        z = torch.relu(self.hid7(z))
        z = torch.relu(self.hid8(z))
        z = self.output(z)
        return z


