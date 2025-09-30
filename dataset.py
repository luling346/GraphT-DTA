import numpy as np
import torch
import dgl

from torch.utils.data import Dataset
# from data_process import *
from dgl import load_graphs


device = torch.device('cuda')

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]


class GTDataset(Dataset):
    def __init__(self, compounds_graph=None, proteins=None, interactions=None):

        self.proteins = load_tensor(proteins, torch.FloatTensor)
        self.interactions = load_tensor(interactions, torch.LongTensor)
        self.compounds_graph, _ = load_graphs(compounds_graph)

        self.compounds_graph = list(self.compounds_graph)
        # self.compound_smiles = load_tensor(compounds_smiles, torch.FloatTensor)

        # assert len(self.compounds) == len(self.proteins) == len(self.interactions)

    def collate(self, sample):
        N = len(sample)
        compounds_graph, proteins, interactions = map(list, zip(*sample))
        compounds_graph = dgl.batch(compounds_graph).to(device)
        # compounds_smiles = torch.stack(compounds_smiles)

        max_length = 1000
        for i in range(len(proteins)):
            if proteins[i].shape[0] < max_length:
                zero = torch.zeros((max_length - proteins[i].shape[0], proteins[i].shape[1]), dtype=torch.long).to(device)
                proteins[i] = torch.cat((proteins[i], zero), dim=0)
                # data_seq = np.concatenate((Seq_onehot, zero), axis=0)
            else:
                proteins[i] = proteins[i][:max_length, :]
        proteins = torch.stack(proteins)

        interactions = torch.tensor(interactions).long().to(device)
        return compounds_graph, proteins, interactions


    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, item):
        return self.compounds_graph[item], self.proteins[item], self.interactions[item]
