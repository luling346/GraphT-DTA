import dgl
import numpy as np
import torch
import pandas as pd
import os.path as osp
from rdkit import Chem

from dgl.data.utils import save_graphs
from scipy import sparse as sp

from word2vec import seq_to_kmers, get_protein_embedding
from gensim.models import Word2Vec

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import warnings
warnings.filterwarnings("ignore")


def data_split_train_val_test(data_root='dataset', data_set='celegans'):

    data_path = osp.join(data_root, data_set, 'processed', 'data.csv')
    data_df = pd.read_csv(data_path)

    # Split data in train:val:test = 8:1:1 with the same random seed as previous study.
    data_shuffle = data_df.sample(frac=1., random_state=1234)
    train_split_idx = int(len(data_shuffle) * 0.8)
    df_train = data_shuffle[:train_split_idx]
    df_val_test = data_shuffle[train_split_idx:]
    val_split_idx = int(len(df_val_test) * 0.5)
    df_val = df_val_test[:val_split_idx]
    df_test = df_val_test[val_split_idx:]

    df_train.to_csv(osp.join(data_root, data_set, 'processed', 'data_train.csv'), index=False)
    df_val.to_csv(osp.join(data_root, data_set, 'processed', 'data_val.csv'), index=False)
    df_test.to_csv(osp.join(data_root, data_set, 'processed', 'data_test.csv'), index=False)

    print(f"{data_set} split done!")
    print("Number of data: ", len(data_df))
    print("Number of train: ", len(df_train))
    print("Number of val: ", len(df_val))
    print("Number of test: ", len(df_test))


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix(scipy_fmt='csr').astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    if EigVec.shape[1] < pos_enc_dim + 1:
        PadVec = np.zeros((EigVec.shape[0], pos_enc_dim + 1 - EigVec.shape[1]), dtype=EigVec.dtype)
        EigVec = np.concatenate((EigVec, PadVec), 1)
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()
    return g


def atom_features(atom, explicit_H=False, use_chirality=True):
    """Generate atom features including atom symbol(17),degree(7),formal charge(1),
    radical electrons(1),hybridization(6),aromatic(1),hydrogen atoms attached(5),Chirality(3)
    """
    symbol = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Si', 'Fe', 'Zn', 'Cu', 'Mn', 'Mo', 'other']  # 17-dim
    degree = [0, 1, 2, 3, 4, 5, 6]  # 7-dim
    hybridizationType = [Chem.rdchem.HybridizationType.SP,
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3,
                              Chem.rdchem.HybridizationType.SP3D,
                              Chem.rdchem.HybridizationType.SP3D2,
                              'other']   # 6-dim
    results = one_of_k_encoding_unk(atom.GetSymbol(),symbol) + \
                  one_of_k_encoding(atom.GetDegree(),degree) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [atom.GetIsAromatic()]  # 17+7+2+6+1=33

    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                      [0, 1, 2, 3, 4])   # 33+5=38
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  # 38+3 =41
    return results


def bond_features(bond, use_chirality=True):
    """Generate bond features including bond type(4), conjugated(1), in ring(1), stereo(4)"""
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats).astype(int)



def smiles_to_graph(smiles, explicit_H=False, use_chirality=True):

    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        raise RuntimeError("SMILES cannot been parsed!")
    g = dgl.DGLGraph()
    # Add nodes
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)

    atom_feats = np.array([atom_features(a, explicit_H=explicit_H) for a in mol.GetAtoms()])
    if use_chirality:
        chiralcenters = Chem.FindMolChiralCenters(mol, force=True, includeUnassigned=True,
                                                  useLegacyImplementation=False)
        chiral_arr = np.zeros([num_atoms, 3])
        for (i, rs) in chiralcenters:
            if rs == 'R':
                chiral_arr[i, 0] = 1
            elif rs == 'S':
                chiral_arr[i, 1] = 1
            else:
                chiral_arr[i, 2] = 1
        atom_feats = np.concatenate([atom_feats, chiral_arr], axis=1)

    g.ndata["atom"] = torch.tensor(atom_feats)

    # Add edges
    src_list = []
    dst_list = []
    bond_feats_all = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        bond_feats = bond_features(bond, use_chirality=use_chirality)
        src_list.extend([u, v])
        dst_list.extend([v, u])
        bond_feats_all.append(bond_feats)
        bond_feats_all.append(bond_feats)

    g.add_edges(src_list, dst_list)

    # g.edata["bond"] = torch.tensor(np.array(bond_feats_all))
    return g


def data_generation(dataset, type):

    compounds, proteins, interactions = [], [], []
    model = Word2Vec.load("word2vec_30.model")

    filename = 'dataset/' + dataset + '/processed/data_' + type + '.csv'
    N = len(open(filename).readlines())
    print(N)

    df = pd.read_csv(filename)
    print(len(list(df)))

    for i, row in df.iterrows():
        print('/'.join(map(str, [i + 1, N - 1])))
        smiles = row['compound_iso_smiles']
        sequence = row['target_sequence']
        interaction = row['affinity']

        compound_graph = smiles_to_graph(smiles)
        compound_graph = laplacian_positional_encoding(compound_graph, pos_enc_dim=8)
        protein_embedding = get_protein_embedding(model, seq_to_kmers(sequence))

        compounds.append(compound_graph)
        proteins.append(protein_embedding)
        interactions.append(np.array([float(interaction)]))

    return compounds, proteins, interactions



if __name__ == '__main__':

    DATASET = "celegans"
    type = 'train'
    # data_split_train_val_test(data_root='dataset', data_set=DATASET)

    compounds, proteins, interactions = data_generation(DATASET, type)

    dir_input = ('dataset/' + DATASET + '/processed/' + type + '/')
    os.makedirs(dir_input, exist_ok=True)

    print(len(proteins))
    dgl.save_graphs(dir_input + 'compounds1.bin', list(compounds))
    # np.save(dir_input + 'proteins.npy', proteins)
    # np.save(dir_input + 'interactions.npy', interactions)

    print('The preprocess of ' + DATASET + ' dataset has finished!')