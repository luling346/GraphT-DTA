import os
import random
import timeit

import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from dataset import *

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, precision_recall_curve, auc
from dgl import load_graphs

from models.net import GTDTInet

#device = torch.device('cuda')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, device, train_loader, optimizer):
    # print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        label = data[-1].to(device)
        compound_graph, protein = data[:-1]
        compound_graph = compound_graph.to(device)
        protein = protein.to(device)
        pred = model(compound_graph, protein)
        loss = criterion(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(model, device, test_loader):
    model.eval()
    test_losses = []

    pred_list = []
    pred_cls_list = []
    label_list = []

    with torch.no_grad():
        for data in test_loader:
            label = data[-1].to(device)
            compound_graph, protein = data[:-1]
            compound_graph = compound_graph.to(device)
            protein = protein.to(device)
            pred = model(compound_graph, protein)
            loss = criterion(pred, label)

            pred_cls = torch.argmax(pred, dim=-1)
            pred_prob = F.softmax(pred, dim=-1)
            pred_prob, indices = torch.max(pred_prob, dim=-1)
            pred_prob[indices == 0] = 1. - pred_prob[indices == 0]

            pred_list.append(pred_prob.view(-1).detach().cpu().numpy())
            pred_cls_list.append(pred_cls.view(-1).detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

    pred = np.concatenate(pred_list, axis=0)
    pred_cls = np.concatenate(pred_cls_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    Accuracy = accuracy_score(label, pred_cls)
    Precision = precision_score(label, pred_cls)
    Recall = recall_score(label, pred_cls)
    AUC = roc_auc_score(label, pred)
    tpr, fpr, _ = precision_recall_curve(label, pred)
    AUPR = auc(fpr, tpr)
    return AUC, AUPR, Accuracy, Precision, Recall


if __name__ == '__main__':

    #Davis
    
    DATASET = "Davis"
    dir_input = ('dataset/' + DATASET + '/processed/')
    # dir_input = ('/2111041014/DTI/GTDTI/DTI/dataset/' + DATASET + '/processed/')
    os.makedirs(dir_input, exist_ok=True)

    batch = 60
    lr = 1e-4
    epochs = 1000

    dataset_train = GTDataset(compounds_graph=dir_input + 'train/compounds1.bin',
                              proteins=dir_input + 'train/proteins',
                              interactions=dir_input + 'train/interactions')
    dataset_val = GTDataset(compounds_graph=dir_input + 'val/compounds1.bin',
                            proteins=dir_input + 'val/proteins',
                            interactions=dir_input + 'val/interactions')
    dataset_test = GTDataset(compounds_graph=dir_input + 'test/compounds1.bin',
                             proteins=dir_input + 'test/proteins',
                             interactions=dir_input + 'test/interactions')

    train_loader = DataLoader(dataset_train, batch_size=batch, shuffle=True, collate_fn=dataset_train.collate, drop_last=True)
    val_loader = DataLoader(dataset_val, batch_size=batch, shuffle=False, collate_fn=dataset_val.collate, drop_last=True)
    test_loader = DataLoader(dataset_test, batch_size=batch, shuffle=False, collate_fn=dataset_test.collate, drop_last=True)

    model = GTDTInet(max_length=1000, compound_graph_dim=128, protein_dim=128, out_dim=2)
    model.to(device)


    start = timeit.default_timer()
    best_auc = 0
    best_epoch = -1

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    Indexes = ('Epoch\t\tTime\t\tAUC\t\tAUPR\t\tAccuracy\t\tPrecision\t\tRecall')

    """Start training."""
    print('Training on ' + DATASET)
    print(Indexes)

    for epoch in range(epochs):
        train(model, device, train_loader, optimizer)
        AUC, AUPR, Accuracy, Precision, Recall = test(model, device, test_loader)
        end = timeit.default_timer()
        time = end - start
        ret = [epoch + 1, round(time, 2), round(AUC, 5), round(AUPR, 5), round(Accuracy, 5), round(Precision, 5), round(Recall, 5)]
        print('\t\t'.join(map(str, ret)))
        if ret[2] > best_auc:
            best_epoch = epoch + 1
            print('AUC improved at epoch ', best_epoch, ';\tbest_AUC:', ret[2])

